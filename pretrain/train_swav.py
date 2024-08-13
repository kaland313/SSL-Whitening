# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
from omegaconf import DictConfig
import torch
from torch import nn

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVPrototypes, SwaVProjectionHead
from lightly.transforms.swav_transform import SwaVTransform

from pretrain.trainer_common import LightlyModel, main_pretrain

class SwaV(LightlyModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Head based on https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#cifar-10
        self.projection_head = SwaVProjectionHead(
            input_dim=self.backbone.num_features,
            hidden_dim=512, 
            output_dim=128,
        )
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        self.criterion = SwaVLoss()

    def setup_transform(self):
        if self.input_size == 32:
            self.transform = SwaVTransform(crop_sizes=[32],
                                           crop_counts=[2],  # 2 crops @ 32x32px
                                           crop_min_scales=[0.14],
                                           cj_strength=0.5,
                                           gaussian_blur=0,
                                           )
        elif self.input_size == 64:
            self.transform = SwaVTransform(crop_sizes=[64],
                                           crop_counts=[2],  # 2 crops @ 64x64px
                                           crop_min_scales=[0.14],
                                           cj_strength=0.5,
                                           gaussian_blur=0,
                                           )                                           
        elif self.input_size == 96:
            self.transform = SwaVTransform(crop_sizes=(96,48), 
                                           crop_min_scales=(0.5, 0.3), 
                                           crop_max_scales=(1.0, 0.5),
                                           )
        else:
            self.transform = SwaVTransform()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        z = nn.functional.normalize(z, dim=1, p=2)
        p = self.prototypes(z)
        return p

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        self.prototypes.normalize()
        views = batch[0]
        multi_crop_features = [self.forward(view.to(self.device)) for view in views]
        high_resolution = multi_crop_features[:2]
        low_resolution = multi_crop_features[2:]
        loss = self.criterion(high_resolution, low_resolution)
        self.log(f"{metric_label}/swav_loss", loss, on_epoch=True)
        return loss

@hydra.main(version_base="1.2", config_path="configs/", config_name="swav.yaml")
def pretrain_swav(cfg: DictConfig):
    main_pretrain(cfg, SwaV)

if __name__ == "__main__":
    pretrain_swav()