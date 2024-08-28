# SSL Whitening

This repository is the official implementation of [*Whitening Consistently Improves Self-Supervised Learning*](https://arxiv.org/abs/2408.07519)

[[arxiv](https://arxiv.org/abs/2408.07519)]

![Architecture](.github/Architecture.svg)

If you use our code or results, please cite our paper and consider giving this repo a :star: :
```
@misc{kalapos2024whiteningconsistentlyimproves,
      title={Whitening Consistently Improves Self-Supervised Learning}, 
      author={András Kalapos and Bálint Gyires-Tóth},
      year={2024},
      eprint={2408.07519},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.07519}, 
}
```

## How to run
For each SSL method, we provide a script to run the training. The scripts are located in the [pretrain](pretrain) folder.

The following pretraining methods are implemented:
- [Barlow Twins](pretrain/train_barlowtwins.py)
- [BYOL](pretrain/train_byol.py)
- [SimCLR](pretrain/train_simclr.py)
- [SwAV](pretrain/train_swav.py)
- [VICReg](pretrain/train_vicreg.py)
- [Supervised](pretrain/train_supervised.py)

E.g. to run BYOL pretraining:


    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python pretrain/train_byol.py


## Setup

We recommend using the provided Docker container to run the code. 

### Option A: Start Docker container and connect to it via ssh: 
1. Create a keypair, copy the public key to the root of this repo and name it `cm-docker.pub`!
2. Run `make ssh`.
3. Connect on port 2233 `ssh root@<hostname> -i <private_key_path> -p 2222`.

To run the container without starting an ssh server, run `make run`.

To customize Docker build and run, edit the [Makefile](Makefile) or the [Dockerfile](Dockerfile).

> [!WARNING]
> `make ssh` and `make run` start the container with the `--rm` flag! Only contents of the `/workspace` persist if the container is stopped (via a simple volume mount)!

### Option B: Install dependencies locally (not tested)

Install the requirements with `pip install -r requirements.txt`.

### Dataset setup
To set the path for the datasets, edit the [Makefile](Makefile)'s `data_path=...` line.

CIFAR-10 and STL-10 download automatically; to set up TinyImageNet, we provide a script: [utils/tiny_imagenet_setup.py](utils/tiny_imagenet_setup.py).


## Copyright, acknowledgements
Whitening is implemented based on [<img src="https://github.githubassets.com/pinned-octocat.svg" style="height:12pt;" /> huangleiBuaa/IterNorm](https://github.com/huangleiBuaa/IterNorm).

Our implementation is based on the  <img src="https://github.githubassets.com/pinned-octocat.svg" style="height:12pt;" /> [Lightly library](https://github.com/lightly-ai/lightly).
