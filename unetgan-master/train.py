#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go. """
import os
import functools
import math
import numpy as np
use_tqdm=False
if use_tqdm:
    from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
####
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PyTorchDatasets import CocoAnimals
from PyTorchDatasets import  FFHQ,Celeba
# Import my stuff
import inception_utils
import utils

from PyTorchDatasets import CocoAnimals, FFHQ, Celeba
from fid_score import calculate_fid_given_paths_or_tensor
from torchvision.datasets import ImageFolder
import pickle
from matplotlib import pyplot as plt
from mixup import CutMix
import gc
import sys
from types import ModuleType, FunctionType
from gc import get_referents


def run(config):

    






















def main():

    # parse command line and run
    parser = unet_utils.prepare_parser()
    config = vars(parser.parse_args())

    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    random_number_string = str(int(np.random.rand()*1000000)) + "_" + config["id"]
    config["stop_it"] = 99999999999999


    if config["debug"]:
        config["save_every"] = 30
        config["sample_every"] = 20
        config["test_every"] = 20
        config["num_epochs"] = 1
        config["stop_it"] = 35
        config["slow_mixup"] = False

    config["num_gpus"] = len(config["gpus"].replace(",",""))

    config["random_number_string"] = random_number_string
    new_root = os.path.join(config["base_root"],random_number_string)
    if not os.path.isdir(new_root):
        os.makedirs(new_root)
        os.makedirs(os.path.join(new_root, "samples"))
        os.makedirs(os.path.join(new_root, "weights"))
        os.makedirs(os.path.join(new_root, "data"))
        os.makedirs(os.path.join(new_root, "logs"))
        print("created ", new_root)
    config["base_root"] = new_root


    keys = sorted(config.keys())
    print("config")
    for k in keys:
        print(str(k).ljust(30,"."), config[k] )



    run(config)


if __name__ == '__main__':
    main()
