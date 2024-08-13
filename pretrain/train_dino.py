# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
import timm

from lightly.data import LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

from pretrain.metrics import compute_contrastive_acc, log_example_inputs

from pretrain.trainer_common import LightlyModelMomentum, main_pretrain


class DINO(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Head dimensions differ from the paper to fit the CIFAR-10 (and are based on https://docs.lightly.ai/self-supervised-learning/examples/dino.html) 
        # projection head is aka the student head
        # The dino projection head doesn't use BatchNorm by default (it uses L2 before the last layer) -> we don't experiment with LayerNorm instead of BatchNorm.
        self.projection_head = DINOProjectionHead(
            input_dim=self.backbone.num_features, hidden_dim=512, bottleneck_dim=64, output_dim=2048, freeze_last_layer=1) 
        # projection head momentum is aka the teacher head
        self.projection_head_momentum = DINOProjectionHead(
            input_dim=self.backbone.num_features, hidden_dim=512, bottleneck_dim=64, output_dim=2048) 
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def setup_transform(self):
        if self.input_size == 32:
            self.transform = DINOTransform(global_crop_size=32,
                                           global_crop_scale=(0.5, 1.0),
                                           n_local_views=0,
                                           cj_strength=0.5,
                                           gaussian_blur=(0, 0, 0),
            )
        elif self.input_size == 64:
            self.transform = DINOTransform(global_crop_size=64,
                                           global_crop_scale=(0.5, 1.0),
                                           local_crop_size=32,
                                           local_crop_scale=(0.3, 0.5),
                                           cj_strength=0.5,
                                           gaussian_blur=(0, 0, 0),
                                           )             
        elif self.input_size == 96:
            self.transform = DINOTransform(global_crop_size=96,
                                           global_crop_scale=(0.5, 1.0),
                                           local_crop_size=48,
                                           local_crop_scale=(0.3, 0.5),
            )
        else:
            self.transform = DINOTransform()

    def forward(self, x):
        # AKA. student forward pass
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def forward_momentum(self, x):
        # AKA. teacher forward pass
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        views = batch[0]
        # views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_momentum(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log(f"{metric_label}/dino_loss", loss, on_epoch=True)
        return loss

    def on_after_backward(self):
        self.projection_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

@hydra.main(version_base="1.2", config_path="configs/", config_name="dino.yaml")
def pretrain_dino(cfg: DictConfig):
    main_pretrain(cfg, DINO)

if __name__ == "__main__":
    pretrain_dino()