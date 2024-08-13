# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import ProjectionHead 
from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform
from lightly.utils.lars import LARS
from lightly.models.utils import get_weight_decay_parameters
from lightly.models.modules.heads import ProjectionHead
import torch.nn as nn
from timm.layers import LayerNorm2d, LayerNorm


from pretrain.trainer_common import LightlyModel, main_pretrain


class BarlowTwins(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)       
        # use a 2-layer projection head for cifar10 as described in the SimSiam paper
        self.projection_head = ProjectionHead(
            [
                (self.backbone.num_features, 2048, nn.BatchNorm1d(2048), nn.ReLU(inplace=True)),
                (2048, 2048, None, None),
            ]
        )
        
        # self.projection_head = BarlowTwinsProjectionHead(self.backbone.num_features, 2048, 2048)
        self.criterion = BarlowTwinsLoss()

    def setup_transform(self):
        # BarlowTwins uses BYOL augmentations.
        self.transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=self.input_size),
            view_2_transform=BYOLView2Transform(input_size=self.input_size)
            )

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        return z

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log(f"{metric_label}/barlowtwins_loss", loss, on_epoch=True)
        return loss

    # def configure_optimizers(self):
    #     # Source: https://github.com/lightly-ai/lightly/blob/e7f13a53e68031ee0ae77124e76401b48381001f/benchmarks/imagenet/resnet50/barlowtwins.py#L18
    #     lr_factor = self.cfg.optimizer.batch_size * self.trainer.world_size / 256

    #     # Don't use weight decay for batch norm, bias parameters, and classification
    #     # head to improve performance.
    #     params, params_no_weight_decay = get_weight_decay_parameters(
    #         [self.backbone, self.projection_head]
    #     )
    #     optim = LARS(
    #         [
    #             {"name": "barlowtwins", "params": params},
    #             {
    #                 "name": "barlowtwins_no_weight_decay",
    #                 "params": params_no_weight_decay,
    #                 "weight_decay": 0.0,
    #                 "lr": 0.0048 * lr_factor,
    #             },
    #         ],
    #         lr=0.2 * lr_factor,
    #         momentum=0.9,
    #         weight_decay=1.5e-6,
    #     )

    #     return optim


@hydra.main(version_base="1.2", config_path="configs/", config_name="barlowtwins.yaml")
def pretrain_barlowtwins(cfg: DictConfig):
    main_pretrain(cfg, BarlowTwins)


if __name__ == "__main__":
    pretrain_barlowtwins()
