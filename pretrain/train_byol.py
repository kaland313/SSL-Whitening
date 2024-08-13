# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

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
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.utils.scheduler import cosine_schedule

from pretrain.metrics import compute_contrastive_acc, log_example_inputs
from pretrain.trainer_common import LightlyModelMomentum, main_pretrain


class BYOL(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.projection_head = BYOLProjectionHead(self.backbone.num_features, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def setup_transform(self):
        self.transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=self.input_size),
            view_2_transform=BYOLView2Transform(input_size=self.input_size)
            )

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        x0, x1 = batch[0]
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log(f"{metric_label}/byol_loss", loss, on_epoch=True)
        return loss


@hydra.main(version_base="1.2", config_path="configs/", config_name="byol.yaml")
def pretrain_byol(cfg: DictConfig):
    main_pretrain(cfg, BYOL)

if __name__ == "__main__":
    pretrain_byol()
