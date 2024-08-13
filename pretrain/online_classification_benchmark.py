# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import knn_predict
from lightly.utils.benchmarking.topk import mean_topk_accuracy


class OnlineLinearClassificationBenckmark:
    def __init__(
        self,
        backbone,
        num_classes,
        dataset_class,
        train_dataset_kwargs,
        val_dataset_kwargs,
        batch_size=256,
        num_workers=8,
        device="cuda",
        topk=(1,5),
    ):
        self.device = device
        self.backbone = backbone
        self.num_classes = num_classes
        self.topk = topk
        self.classifier = nn.Linear(backbone.num_features, num_classes).to(self.device)
        for p in self.classifier.parameters():
            p.requires_grad = True
        self.optimizer = optim.Adam(self.classifier.parameters())

        # Dataset & Dataloader setup
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ]
        )

        self.train_dataset = dataset_class(**train_dataset_kwargs, transform=transform)
        self.val_dataset = dataset_class(**val_dataset_kwargs, transform=transform)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

    def run_online_classification_benchmarks(self, num_epochs=2):
        """ Runs the online linear classification benchmark.

        Args:
            num_epochs (int, optional): Trains the linear classification layer for num_epochs epochs. Defaults to 2.

        Returns:
            dict: benchmark results
        """
        train_features, train_targets = self.compute_features(self.train_dataloader)
        val_features, val_targets = self.compute_features(self.val_dataloader)

        for epoch in range(num_epochs):
            self.fit_lin_classifier(train_features, train_targets)
            lin_results_dict = self.evaluate_lin_classifier(val_features, val_targets)
            print(f"Benchmark Epoch {epoch+1}/{num_epochs}, Lin Accuracy: ", *[f"{k}: {v*100:.2f}%" for k, v in lin_results_dict.items()])

        knn_results_dict = self.evaluate_knn_classifier(
            train_features, train_targets, val_features, val_targets
        )

        results_dict = {**lin_results_dict, **knn_results_dict}
        print(f"Accuracy: ", *[f"\n  {k}: {v*100:.2f}%" for k, v in results_dict.items()])
        return results_dict
    
    @torch.no_grad()
    def compute_features(self, dataloader):
        """
        Compute features for the whole dataset.
        """
        features = []
        targets = []
        for batch in tqdm(dataloader, desc="Computing features"):
            inputs, targets_batch = batch
            inputs = inputs.to(self.device)
            targets_batch = targets_batch.to(self.device)

            representations = self.backbone(inputs)
            if len(representations.shape) > 2:
                # if we get pre-pooling feature maps, pool them.
                representations = torch.flatten(
                    F.adaptive_avg_pool2d(representations, 1), start_dim=1
                )
            features.append(representations)
            targets.append(targets_batch)
        return features, targets
    
    def fit_lin_classifier(self, train_features, train_targets):
        for batch in zip(train_features, train_targets):
            features_batch, targets_batch = batch
            # Classifier forward pass and optimization
            with torch.enable_grad():
                # If we call online_linear_classification_benchmark from a lightning module's on_validation_epoch_end, 
                # gradient computation is disabled by default. (check it with torch.is_grad_enabled())
                # For training the linear classifier we need to enable it again.
                self.optimizer.zero_grad()
                outputs = self.classifier(features_batch)
                loss = nn.CrossEntropyLoss()(outputs, targets_batch)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def evaluate_lin_classifier(self, val_features, val_labels):
        val_features_tensor = torch.cat(val_features, dim=0)
        val_labels_tensor = torch.cat(val_labels, dim=0)

        outputs = self.classifier(val_features_tensor)
        
        _, predicted_classes = outputs.topk(max(self.topk))

        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=val_labels_tensor, k=self.topk
        )
        results_dict = {f"lin_top{k}": acc for k, acc in topk.items()}
        return results_dict
    

    def evaluate_knn_classifier(self, feature_bank, label_bank, val_features, val_labels, k=200, t=0.1):
        feature_bank_tensor = torch.cat(feature_bank, dim=0)
        label_bank_tensor = torch.cat(label_bank, dim=0)
        val_features_tensor = torch.cat(val_features, dim=0)
        val_labels_tensor = torch.cat(val_labels, dim=0)

        feature_bank_tensor = F.normalize(feature_bank_tensor, dim=1).T
        val_features_tensor = F.normalize(val_features_tensor, dim=1)

        predicted_classes = knn_predict(
            feature=val_features_tensor,
            feature_bank=feature_bank_tensor,
            feature_labels=label_bank_tensor,
            num_classes=self.num_classes,
            knn_k=k,
            knn_t=t,
        )
        topk = mean_topk_accuracy(
            predicted_classes=predicted_classes, targets=val_labels_tensor, k=self.topk
        )
        results_dict = {f"knn_top{k}": acc for k, acc in topk.items()}

        return results_dict


def test_online_classification():
    from pretrain.trainer_common import LightlyModel

    cfg = DictConfig(
        {
            "data": {
                "dataset_name": "cifar10",
                "num_workers": 8,
            },
            "backbone": {
                "name": "resnet18",
                "pretrained_weights": "imagenet",
                "kwargs": {},
            },
            "optimizer": {
                "lr": None,
            },
        }
    )

    model = LightlyModel(cfg)
    model.setup("validate")
    model.online_classifier.device = "cuda"
    model.backbone.to(model.online_classifier.device)
    model.online_classifier.run_online_classification_benchmarks()


if __name__ == "__main__":
    test_online_classification()