import argparse
import os, sys
import warnings
import pandas as pd
import time
import numpy as np
from torch.utils import data
import yaml, csv
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.utils as utils
import lib.odenvp as odenvp
from lib.datasets import CelebAHQ, Imagenet64

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import append_regularization_keys_header, append_regularization_csv_dict

import dist_utils
from dist_utils import env_world_size, env_rank
from torch.utils.data.distributed import DistributedSampler

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'adaptive_heun', 'bosh3']

def get_parser():
    parser = argparse.ArgumentParser("Continuous Normalizing Flow")
    parser.add_argument("--datadir", default="./data/")
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church', 'celebahq', 'imagenet64'], 
            type=str, default="mnist")
    parser.add_argument("--dims", type=str, default="64,64,64")
    parser.add_argument("--strides", type=str, default="1,1,1,1")
    parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')

    parser.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat"]
    )
    parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    parser.add_argument(
        "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"]
    )
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--atol', type=float, default=1e-5, help='only for adaptive solvers')
    parser.add_argument('--rtol', type=float, default=1e-5,  help='only for adaptive solvers')
    parser.add_argument('--step_size', type=float, default=0.25, help='only for fixed step size solvers')
    parser.add_argument('--first_step', type=float, default=0.166667, help='only for adaptive solvers')

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)
    parser.add_argument('--test_step_size', type=float, default=None)
    parser.add_argument('--test_first_step', type=float, default=None)

    parser.add_argument("--imagesize", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument('--time_length', type=float, default=1.0)
    parser.add_argument('--train_T', type=eval, default=False)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument(
        "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
    )
    parser.add_argument("--test_batch_size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=float, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.)

    parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--div_samples',type=int, default=1)
    parser.add_argument('--squeeze_first', type=eval, default=False, choices=[True, False])
    parser.add_argument('--zero_last', type=eval, default=True, choices=[True, False])
    parser.add_argument('--seed', type=int, default=42)

    # Regularizations
    parser.add_argument('--kinetic-energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--jacobian-norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total-deriv', type=float, default=None, help="int_t ||df/dt||^2")
    parser.add_argument('--directional-penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    parser.add_argument(
        "--max_grad_norm", type=float, default=np.inf,
        help="Max norm of graidents"
    )

    parser.add_argument("--resume", type=str, default=None, help='path to saved check point')
    parser.add_argument("--save", type=str, default="experiments/cnf")
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument('--validate', type=eval, default=False, choices=[True, False])

    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')

    #parser.add_argument('--skip-auto-shutdown', action='store_true',
    #                    help='Shutdown instance at the end of training or failure')
    #parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
    #                    help='how long to wait until shutting down on success')
    #parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
    #                    help='how long to wait before shutting down on error')

    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
torch.manual_seed(args.seed)
nvals = 2**args.nbits

# Only want master rank logging
is_master = (not args.distributed) or (dist_utils.env_rank()==0)
is_rank0 = args.local_rank == 0
write_log = is_rank0 and is_master


def add_noise(x, nbits=8):
    if nbits<8:
        x = x // (2**(8-nbits))
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
    else:
        noise = 1/2
    return x.add_(noise).div_(2**nbits)

def shift(x, nbits=8):
    if nbits<8:
        x = x // (2**(8-nbits))

    return x.add_(1/2).div_(2**nbits)

def unshift(x, nbits=8):
    return x.add_(-1/(2**(nbits+1)))


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size)])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root=args.datadir, train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root=args.datadir, train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root=args.datadir, split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root=args.datadir, split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root=args.datadir, train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
            ]), download=True
        )
        test_set = dset.CIFAR10(root=args.datadir, train=False, transform=None, download=True)
    elif args.data == 'celebahq':
        im_dim = 3
        im_size = 256 if args.imagesize is None else args.imagesize
        train_set = CelebAHQ(
            train=True, root=args.datadir, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
            ])
        )
        test_set = CelebAHQ(
            train=False, root=args.datadir,  transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
            ])
        )
    elif args.data == 'imagenet64':
        im_dim = 3
        if args.imagesize != 64:
            args.imagesize = 64
        im_size = 64
        train_set = Imagenet64(train=True, root=args.datadir)
        test_set = Imagenet64(train=False, root=args.datadir)
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
            ])
        )
        test_set = dset.LSUN(
            'data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
            ])
        )
    data_shape = (im_dim, im_size, im_size)

    def fast_collate(batch):

        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
        w = imgs[0].size[0]
        h = imgs[0].size[1]

        tensor = torch.zeros( (len(imgs), im_dim, im_size, im_size), dtype=torch.uint8 )
        for i, img in enumerate(imgs):
            nump_array = np.asarray(img, dtype=np.uint8)
            tens = torch.from_numpy(nump_array)
            if(nump_array.ndim < 3):
                nump_array = np.expand_dims(nump_array, axis=-1)
            nump_array = np.rollaxis(nump_array, 2)
            tensor[i] += torch.from_numpy(nump_array)

        return tensor, targets

    train_sampler = (DistributedSampler(train_set,
        num_replicas=env_world_size(), rank=env_rank()) if args.distributed
        else None)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, #shuffle=True,
        num_workers=args.nworkers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate
    )

    test_sampler = (DistributedSampler(test_set,
        num_replicas=env_world_size(), rank=env_rank(), shuffle=False) if args.distributed
        else None)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, #shuffle=False,
        num_workers=args.nworkers, pin_memory=True, sampler=test_sampler, collate_fn=fast_collate
    )

    return train_loader, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp, reg_states = model(x, zero)  # run model forward

    reg_states = tuple(torch.mean(rs) for rs in reg_states)

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(nvals)) / np.log(2)

    return bits_per_dim, (x, z), reg_states


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    model = odenvp.ODENVP(
        (args.batch_size, *data_shape),
        n_blocks=args.num_blocks,
        intermediate_dims=hidden_dims,
        div_samples=args.div_samples,
        strides=strides,
        squeeze_first=args.squeeze_first,
        nonlinearity=args.nonlinearity,
        layer_type=args.layer_type,
        zero_last=args.zero_last,
        alpha=args.alpha,
        cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
    )

    return model



if __name__ == "__main__":



    # get deivce
    device = torch.device("cuda:%d"%torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)



    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)


    # optimizer
    if args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                nesterov=False)


        
    dummy_input = torch.randn(300, 1, 28, 28)
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "cnf.onnx", verbose=False)
    print(data_shape)
    print(len(train_loader))