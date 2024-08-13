# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
from typing import Callable
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn as nn
import timm
import wandb
import matplotlib.pyplot as plt

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from pretrain.metrics import contrastive_acc_eval, log_example_inputs, eval_feature_descriptors
from pretrain.online_classification_benchmark import OnlineLinearClassificationBenckmark
from timm.layers import LayerNorm
from models.iterative_normalization import IterNormBackBoneWrapper
import utils


class LightlyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()  # save cfg to self.hparams
        self.cfg = cfg
        self.lr = cfg.optimizer.lr
        if cfg.backbone.name.startswith("lightly-resnet"):
            name = cfg.backbone.name.replace("lightly-resnet", "resnet-")
            from lightly.models import ResNetGenerator
            resnet = ResNetGenerator(name=name)
            self.backbone = nn.Sequential(
                *list(resnet.children())[:-1], nn.AdaptiveAvgPool2d(1)
            )
            self.backbone.__dict__["num_features"] = self.backbone(torch.zeros(1,3,32,32)).shape[1]

        else:
            backbone_kwargs = dict(cfg.backbone.kwargs)
            if cfg.backbone.get("use_batch_norm", False):
                backbone_kwargs["norm_layer"] = torch.nn.BatchNorm2d
                # if we use a custom norm layer, conv_mlp must be set to true= dict(norm_layer=torch.nn.BatchNorm2d, conv_mlp=True)
                backbone_kwargs["conv_mlp"] = True 
            self.backbone = timm.create_model(
                cfg.backbone.name,
                pretrained=cfg.backbone.pretrained_weights == "imagenet",
                num_classes=0,
                **backbone_kwargs,
            )
        if cfg.get("use_iternorm", False):
            self.backbone = IterNormBackBoneWrapper(self.backbone)

        self.train_img_paths = (
            []
        )  # only used for contrastive acce eval if trainer.overfit_batches > 0
        self.val_img_paths = (
            []
        )  # only used for contrastive acce eval if trainer.overfit_batches > 0

        self.projection_head = None
        self.criterion = None

    def forward(self, x):
        """Implment forward step for each method!

        Args:
            x: a minibatch of augmented input images
        """
        raise NotImplemented

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        """Implment train_val step for each method!
        log loss using self.log(f"{metric_label}/loss", loss, on_epoch=True)
        """
        raise NotImplemented
    
    def setup_transform(self):
        """ Set sef.transform to a lightly transform by oveerriding this method.
            Use sef.input_size to set the input_size of the transform.
        """
        # We set self.transform to an invalid value to allow this function to be called, but if it's not overriden, we raise an error
        # self.setup calls this method, therefore this hack to allows this class to be instantiated without having to override this method
        self.transform = -1

    def training_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx)
        if self.trainer.overfit_batches > 0:
            self.train_img_paths.extend(batch[2])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.train_val_step(batch, batch_idx, metric_label="val_metrics")
        if batch_idx == 0 and self.trainer.sanity_checking:
            views = self.get_views_to_log_from_batch(batch)
            log_example_inputs(views, log_label="val")
        if self.trainer.overfit_batches > 0:
            self.val_img_paths.extend(batch[2])
        return loss

    def get_views_to_log_from_batch(self, batch):
        # a batch in lightly is a tuple: inputs, targets, filepaths. Views are in batch[0]
        # Override this if the transforms doewsn't return multiple views in inputs
        return batch[0]
    
    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            overfit_batches = self.trainer.overfit_batches
            if self.current_epoch % 10 == 9:
                train_filepaths = self.train_img_paths if overfit_batches > 0 else None
                train_contrastive_acc = self.contrastive_acc_eval(
                    self.dataset_class(**self.train_dataset_kwargs), train_filepaths, 
                )
                self.log("train_metrics/contrastive_acc", train_contrastive_acc, batch_size=64)
            if self.current_epoch % 5 == 0:
                val_filepaths = self.val_img_paths if overfit_batches > 0 else None
                val_contrastive_acc = self.contrastive_acc_eval(
                    self.dataset_class(**self.val_dataset_kwargs), val_filepaths,
                )
                self.log("val_metrics/contrastive_acc", val_contrastive_acc, batch_size=64)
                
                feature_descriptors = self.eval_feature_descriptors(
                    self.dataset_class(**self.val_dataset_kwargs)
                )
                self.log_dict({f"val_metrics/{k}": v for k, v in feature_descriptors.items()})

                benchmark_results_dict = self.online_classifier.run_online_classification_benchmarks()
                self.log_dict({f"val_metrics/{k}": v for k, v in benchmark_results_dict.items()})

    def contrastive_acc_eval(self, dataset, file_paths=None):
        """Override this to customize contrastive acc eval
        """
        return contrastive_acc_eval(self.backbone, dataset, input_size=self.input_size)
    
    def eval_feature_descriptors(self, dataset):
        """Override this to customize feature descriptor eval
        """
        return eval_feature_descriptors(
            self.backbone,
            dataset,
            cfg_name=self.cfg.name,
            current_epoch=self.current_epoch,
        )


    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optim

    def setup(self, stage: str) -> None:
        dataset_classes = {
            "cifar10": torchvision.datasets.CIFAR10,
            "stl10": torchvision.datasets.STL10,
            "tiny-imagenet": torchvision.datasets.ImageFolder,
        }
        train_dataset_kwargs = {
            "cifar10": dict(root="/data/cifar10", download=True),
            "stl10": dict(root="/data/stl10", download=True, split='train+unlabeled'),
            "tiny-imagenet": dict(root="/data/tiny-imagenet-200/train")
        }
        val_dataset_kwargs = {
            "cifar10": dict(root="/data/cifar10", train=False),
            "stl10": dict(root="/data/stl10", split='test'),
            "tiny-imagenet": dict(root="/data/tiny-imagenet-200/val")
        }
        input_sizes = {
            "cifar10": 32,
            "stl10":  96,
            "tiny-imagenet": 64,
        }
        num_classes = {
            "cifar10": 10,
            "stl10":  10,
            "tiny-imagenet": 200,
        }
        self.dataset_class = dataset_classes[self.cfg.data.dataset_name]
        self.train_dataset_kwargs = train_dataset_kwargs[self.cfg.data.dataset_name]
        self.val_dataset_kwargs = val_dataset_kwargs[self.cfg.data.dataset_name]
        self.input_size = input_sizes[self.cfg.data.dataset_name]
        self.num_classes = num_classes[self.cfg.data.dataset_name]

        # Setup self.transform
        self.setup_transform()

        self.train_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.train_dataset_kwargs),
            transform=self.transform
        )
        self.val_dataset = LightlyDataset.from_torch_dataset(
            self.dataset_class(**self.val_dataset_kwargs),
            transform=self.transform
        )

        lin_benchmark_train_kwargs = self.train_dataset_kwargs.copy()
        if self.cfg.data.dataset_name == "stl10":
            lin_benchmark_train_kwargs["split"] = "train"
        self.online_classifier = OnlineLinearClassificationBenckmark(
            backbone=self.backbone,
            num_classes=self.num_classes, 
            dataset_class=self.dataset_class,
            train_dataset_kwargs = lin_benchmark_train_kwargs, 
            val_dataset_kwargs=self.val_dataset_kwargs, 
            num_workers=self.cfg.data.num_workers,
        )

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cfg.data.num_workers,
        )
        return dataloader


class LightlyModelMomentum(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        deactivate_requires_grad(self.backbone_momentum)

        self.projection_head_momentum = None

    def forward_momentum(self, x):
        raise NotImplemented
    
    def training_step(self, batch, batch_idx):
        momentum = cosine_schedule(self.current_epoch, self.cfg.trainer.max_epochs, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        return super().training_step(batch, batch_idx)


def main_pretrain(cfg: DictConfig, lightly_model: LightlyModel):
    print("Running on:", os.environ.get("HOSTNAME", "docker"), flush=True)
    os.system("nvidia-smi")

    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    OmegaConf.set_struct(cfg, False)

    cfg.artifacts_root += "_" + cfg.data.dataset_name

    flat_config = utils.flatten_dict(cfg)
    cfg.name = cfg.name.replace("_{backbone_use_batch_norm}", "_BN" if cfg.backbone.use_batch_norm else "")
    cfg.name = cfg.name.format(**flat_config)

    pl.seed_everything(cfg.seed)

    model = lightly_model(cfg)

    if cfg.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            name=cfg.name, project="SSL-Backbones", save_dir="artifacts",
            group=cfg.get("wandb_group", None),
        )
        # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    root_dir = os.path.abspath(os.path.join(cfg.artifacts_root, cfg.name))
    version = utils.get_next_version(root_dir)
    ckpt_dir = os.path.join(root_dir, f"version_{version}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Checkpoint dir:", ckpt_dir, flush=True)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,  # False to reduce disk load from constant checkpointing
        save_top_k=1,
        monitor="val_metrics/contrastive_acc",
        mode="max",
    )

    trainer = pl.Trainer(
        logger=[wandb_logger] if cfg.wandb else False, 
        callbacks=[checkpoint], 
        **cfg.trainer,
        auto_lr_find=True
    )
    trainer.fit(model=model)

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, max_lr=10)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig(model.__class__.__name__ + ".png")


if __name__ == "__main__":
    main_pretrain(LightlyModel)
