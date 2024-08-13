# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
from lightly.transforms import SimCLRTransform
from lightly.data import LightlyDataset
from tabulate import tabulate


from pretrain import (
    BarlowTwins, BYOL, SimCLR, Supervised, SwaV, VICReg
)


IMAGENET_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
OUT_PATH = "feature_tsv"

model_cls_dict = {
    "barlowtwins": BarlowTwins,
    "byol": BYOL,
    "simclr": SimCLR,
    "supervised": Supervised,
    "swav": SwaV,
    "vicreg": VICReg,
}

def setup_dataset(input_size = 32, batch_size = 64):
    transform = SimCLRTransform(input_size=input_size)
    dataset = torchvision.datasets.CIFAR10(root="/data/cifar10", train=False, download=True)
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
    )  
    return dataloader
    
def compute_features(dataloader, backbone):
    backbone.eval()
    features0 = []
    features1 = []
    labels = []
    for batch in dataloader:
        (x0, x1), label = batch[0], batch[1]
        x0 = x0.to("cuda")
        x1 = x1.to("cuda")
        z0 = backbone(x0)
        z1 = backbone(x1)
        if len(z0.shape) > 2:
            # if we get pre-pooling feature maps, pool them. 
            # compute_contrastive_acc expects (batch,features) shaped tensors
            z0 = torch.flatten(F.adaptive_avg_pool2d(z0, 1), start_dim=1)
            z1 = torch.flatten(F.adaptive_avg_pool2d(z1, 1), start_dim=1)
        features0.append(z0.detach().cpu())
        features1.append(z1.detach().cpu())
        labels.append(label.detach().cpu())

    features0 = torch.cat(features0, dim=0)
    features1 = torch.cat(features1, dim=0)
    labels = torch.cat(labels, dim=0)
    return features0, features1, labels

def feature_cosine_dist(z0, z1):
    return F.cosine_similarity(z0, z1, dim=-1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for model_path in glob("artifacts/pretrain_lightly/**/last.ckpt", recursive=True):
        if "dino" in model_path.lower() or "swav" in model_path.lower():
            continue
        if "cifar10" not in model_path:
            continue
        if "_2D" in model_path:
            continue
        state_dict = torch.load(model_path, map_location=device)
        epoch = state_dict["epoch"]
        if epoch < 100:
            continue
        models.append(model_path)
        print(model_path[len("artifacts/pretrain_lightly/"):], epoch)

    os.makedirs(OUT_PATH, exist_ok=True)

    results = []
    for model_path in tqdm(models):
        model_str = model_path.split('/')[2].split('_')[0]
        model_cls = model_cls_dict[model_str]
        try:
            backbone = model_cls.load_from_checkpoint(model_path).backbone
        except Exception as e:
            print(e)
            continue
        backbone = backbone.to(device)
        dataloader = setup_dataset()
        features0, features1, labels = compute_features(dataloader, backbone)
        

        # tensorboard_projector(features, labels)

        model_name = model_path.split('/')[3]
        model_version = model_path.split('/')[-2]
        out_path = os.path.join(OUT_PATH, model_name + "_" + model_version +  '_last')
        # os.makedirs(out_path, exist_ok=True)
        # np.savetxt(os.path.join(out_path, 'features0.tsv'), features0.numpy(), delimiter='\t')
        # np.savetxt(os.path.join(out_path, 'features1.tsv'), features1.numpy(), delimiter='\t')
        # np.savetxt(os.path.join(out_path, 'labels.tsv'), labels.numpy(), delimiter='\t')        

        cosine_dist = feature_cosine_dist(features0, features1)
        results.append((model_name + "_" + model_version, cosine_dist))

    print(tabulate(results, headers=["model", "cosine_dist"], tablefmt="pipe"))

