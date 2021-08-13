import argparse
import os, sys
import functools
import warnings
import pandas as pd
import time
import numpy as np
import yaml, csv
import shutil
import math
import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as distributed
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from u_net.PyTorchDatasets import  Celeba

import u_net.inception_utils as inception_utils

import lib.layers as layers
import lib.utils as utils
import u_net.utils as unet_utils
import lib.odenvp as odenvp
from lib.datasets import CelebAHQ, Imagenet64

from u_net.fid_score import calculate_fid_given_paths_or_tensor
import pickle
from matplotlib import pyplot as plt
from u_net.mixup import CutMix
import gc
from types import ModuleType, FunctionType
from gc import get_referents


from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import append_regularization_keys_header, append_regularization_csv_dict

import dist_utils
from dist_utils import env_world_size, env_rank
from torch.utils.data.distributed import DistributedSampler

import u_net.unet_d as unet_d

#-----------------------------

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def add_noise(x, nbits=8, add_noise=True): ##** What datatype is x input?
    if nbits<8:
        x = x // (2**(8-nbits)) #Divide by powers of 2, higher powers if bits farther from 8
    if add_noise:
        noise = x.new().resize_as_(x).uniform_()
    else:
        noise = 1/2
    return x.add_(noise).div_(2**nbits) ##** Divide by 2^8 ?!?

def shift(x, nbits=8): ##**
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
    
    data_shape = (im_dim, im_size, im_size)

    def fast_collate(batch): ##** 

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

def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size
            if args.first_step is not None:
                module.solver_options['first_step'] = args.first_step

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol
            if args.test_step_size is not None:
                module.test_solver_options['step_size'] = args.test_step_size
            if args.test_first_step is not None:
                module.test_solver_options['first_step'] = args.test_first_step


    model.apply(_set)



def run(config,args):

    # Only want master rank logging
    is_master = (not args.distributed) or (dist_utils.env_rank()==0)
    is_rank0 = args.local_rank == 0
    write_log = is_rank0 and is_master

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    nvals = 2**args.nbits

    from u_net import train_fns

    config['resolution'] = 128  ##utils.imsize_dict[config['dataset']]
    print("RESOLUTION: ",config['resolution'])
    config['n_classes'] = 1
    config['G_activation'] = nn.ReLU(inplace=False) ##utils.activation_dict[config['G_nl']]
    config['D_activation'] = unet_utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = unet_utils.update_config_roots(config)

    ############################*************************** Need to check this if i use parallel
    device = torch.device("cuda:%d"%torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

    # Prepare root folders if necessary
    unet_utils.prepare_root(config)

    # Import the model--
    model = unet_d
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else unet_utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)
    print("::: weights saved at ", '/'.join([config['weights_root'],experiment_name]) ) ##lg
    
    better_logger = utils.get_logger(logpath=os.path.join(config['logs_root'],'better_logger'),filepath=os.path.abspath(__file__))
    better_logger.info(args)

    better_trainlog = os.path.join(config['logs_root'],'training.csv')
    better_testlog = os.path.join(config['logs_root'],'test.csv')

    traincolumns = ['itr'] ##lg
    testcolumns = ['epoch'] ##lg

    if not args.resume:
        with open(better_trainlog,'w') as f:
            csvlogger = csv.DictWriter(f, traincolumns)
            csvlogger.writeheader()
        with open(better_testlog,'w') as f:
            csvlogger = csv.DictWriter(f, testcolumns)
            csvlogger.writeheader()

    # Next, build the model
    keys = sorted(config.keys())
    for k in keys:
        better_logger.info(f"{k}: {config[k]}")

    

    #-----------------------------------------------------------------------------
    G = model.Generator(args,config).to(device)
    set_cnf_options(args,G)
    #=============================================================================

    D = model.Unet_Discriminator(**config).to(device)

    G_ema, ema = None, None

    GD = model.G_D(G, D, config)

    better_logger.info(G)
    better_logger.info(D)
    better_logger.info('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

    # Prepare noise and randomly sampled label arrays Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = int(G_batch_size*config["num_G_accumulations"])

    z_, y_ = unet_utils.prepare_z_y(G_batch_size, G.dim_z, device=device)
    
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0,'best_FID': 999999,'config': config}
    
    if config['parallel']:
        GD = nn.DataParallel(GD)

    if config['resume']:
        better_logger.info('Loading weights...')
        if config["epoch_id"] !="":
            epoch_id = config["epoch_id"]

        try:
            print("LOADING EMA")
            unet_utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)
        except:
            print("Ema weight wasn't found, copying G weights to G_ema instead")
            unet_utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                             None)
            G_ema.load_state_dict(G.state_dict())

        better_logger.info("loaded weigths")
    
    # Prepare loggers for stats; metrics holds test metrics, lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    better_logger.info('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = unet_utils.MetricsLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
    better_logger.info('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = unet_utils.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
    
    
    # Write metadata
    unet_utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

    if config["dataset"]=="celeba128":
        root =  config["data_folder"] #
        root_perm =  config["data_folder"]
        transform = transforms.Compose(
            [
                transforms.Scale(config["resolution"]),
                transforms.CenterCrop(config["resolution"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        batch_size = config['batch_size']
        dataset = Celeba(root = root, transform = transform, batch_size = batch_size*config["num_D_accumulations"], imsize = config["resolution"])
        data_loader = DataLoader(dataset, batch_size, shuffle = True, drop_last = True)
        loaders = [data_loader]


    better_logger.info(f"Loaded {config['dataset']}")
    inception_metrics_dict = {"fid":[],"is_mean": [], "is_std": []}


    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'],config['parallel'], config['no_fid'], use_torch=False)

    # Prepare data; the Discriminator's batch size is all that needs to be passed to the dataloader, as G doesn't require dataloading. Note
    # that at every loader iteration we pass in enough data to complete a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps'] * config['num_D_accumulations'])

    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = unet_utils.prepare_z_y(G_batch_size, G.dim_z, device=device)
    fixed_z.sample_()
    fixed_y.sample_()

    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()

    # Prepare Sample function for use with inception metrics
    sample = functools.partial(unet_utils.sample,
                          G=(G_ema if config['ema'] and config['use_ema']
                             else G),
                          z_=z_, y_=y_, config=config)

    if config["debug"]:
        loss_steps = 10
    else:
        loss_steps = 100

    better_logger.info(G)
    better_logger.info(D)
    better_logger.info("Number of trainable parameters in Generator CNF: {}".format(count_parameters(G)))
    better_logger.info("Number of trainable parameters in U-Net Discriminator: {}".format(count_parameters(D)))
    better_logger.info('Iters per train epoch: {}'.format(len(data_loader)))
    ##better_logger.info('Iters per test: {}'.format(len(test_loader)))

    better_logger.info('Beginning training at epoch {}...'.format(state_dict['epoch']))


    # Train for specified number of epochs, although we mostly track G iterations.
    warmup_epochs = config["warmup_epochs"]


    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config["progress_bar"]:
            if config['pbar'] == 'mine':
                pbar = unet_utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta',better_logger=better_logger)
            else:
                pbar = tqdm(loaders[0])
        else:
            pbar = loaders[0]

        target_map = None



        for i, batch_data in enumerate(pbar):

            with open(better_trainlog,'a') as f:
                csvlogger = csv.DictWriter(f,[])
                x = batch_data[0]
                y = batch_data[1]
                #H = batch_data[2]

                # Increment the iteration counter
                state_dict['itr'] += 1
                if config["debug"] and state_dict['itr']>config["stop_it"]:
                    better_logger.info("code didn't break :)")
                    break
                # Make sure G and D are in training mode, just in case they got set to eval For D, which typically doesn't have BN, this shouldn't
                # matter much.
                G.train()
                D.train()
                if config['ema']:
                    G_ema.train()
                
                x, y = x.to(device), y.to(device).view(-1)
                x.requires_grad = False
                y.requires_grad = False

                if config["unet_mixup"]:
                    # Here we load cutmix masks for every image in the batch
                    n_mixed = int(x.size(0)/config["num_D_accumulations"])
                    target_map = torch.cat([CutMix(config["resolution"]).cuda().view(1,1,config["resolution"],config["resolution"]) for _ in range(n_mixed) ],dim=0)


                if config["slow_mixup"] and config["full_batch_mixup"]:
                    # r_mixup is the chance that we select a mixed batch instead of
                    # a normal batch. This only happens in the setting full_batch_mixup.
                    # Otherwise the mixed loss is calculated on top of the normal batch.
                    r_mixup = 0.5 * min(1.0, state_dict["epoch"]/warmup_epochs) # r is at most 50%, after reaching warmup_epochs
                elif not config["slow_mixup"] and config["full_batch_mixup"]:
                    r_mixup = 0.5
                else:
                    r_mixup = 0.0

                metrics = train(x, y, state_dict["epoch"], batch_size , target_map = target_map, r_mixup = r_mixup)


                if (i+1)%200==0:
                    # print this just to have some peace of mind that the model is training
                    better_logger.info(f"alive and well at {state_dict['itr']}")

                if (i+1)%20==0:
                    #try:
                    train_log.log(itr=int(state_dict['itr']),csvlog=csvlogger, **metrics)
                    #except:
                    #    print("ouch")
                
                # Every sv_log_interval, log singular values
                if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):

                    train_log.log(itr=int(state_dict['itr']),csvlog=None,
                                **{**unet_utils.get_SVs(G, 'G'), **unet_utils.get_SVs(D, 'D')})

            
                # Save weights and copies as configured at specified interval
                #if not (state_dict['itr'] % config['save_every']):
                if (i+1)%config['save_every']==0:

                    if config['G_eval_mode']:
                        better_logger.info('Switchin G to eval mode...')
                        G.eval()
                        if config['ema']:
                            G_ema.eval()
                        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                        state_dict, config, experiment_name, sample_only=False)
                
                go_ahead_and_sample = ((i+1) % config['sample_every'])==0 

                if go_ahead_and_sample:

                    if config['G_eval_mode']:
                        better_logger.info('Switchin G to eval mode...')
                        G.eval()
                        if config['ema']:
                            G_ema.eval()

                        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                        state_dict, config, experiment_name, sample_only=True)


                        with torch.no_grad():
                            real_batch = dataset.fixed_batch()
                        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                        state_dict, config, experiment_name, sample_only=True, use_real = True, real_batch = real_batch)
                        
                        # also, visualize mixed images and the decoder predicitions
                        if config["unet_mixup"]:
                            with torch.no_grad():

                                n = int(min(target_map.size(0), fixed_z.size(0)/2))
                                which_G = G_ema if config['ema'] and config['use_ema'] else G
                                #unet_utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                #                                            z_, y_, config['n_classes'],
                                #                                            config['num_standing_accumulations'])

                                real_batch = dataset.fixed_batch()
                                fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z[:n], which_G.shared(fixed_z[:n]))) #####shouldnt that be fixed_y?

                                mixed = target_map[:n]*real_batch[:n]+(1-target_map[:n])*fixed_Gz
                                train_fns.save_and_sample(G, D, G_ema, z_[:n], y_[:n], fixed_z[:n], fixed_y[:n],
                                            state_dict, config, experiment_name+"_mix", sample_only=True, use_real = True, real_batch = mixed, mixed=True, target_map = target_map[:n])
                
            # Test every specified interval
            if ((i+1) % config['test_every'])==0:
            #if state_dict['itr'] % 100 == 0:
                if config['G_eval_mode']:
                  better_logger.info('Switchin G to eval mode...')

                is_mean, is_std , fid = train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics , experiment_name, test_log, moments = "train")
                ###
                #  Here, the bn statistics are updated
                ###
                if  config['accumulate_stats']:
                    print("accumulate stats")
                    unet_utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                                                 z_, y_, config['n_classes'], config['num_standing_accumulations'])

                inception_metrics_dict["is_mean"].append((state_dict['itr'] , is_mean ) )
                inception_metrics_dict["is_std"].append((state_dict['itr'] , is_std ) )
                inception_metrics_dict["fid"].append((state_dict['itr'] , fid ) )

            if (i + 1) % loss_steps == 0:
                with open(os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p"), "wb") as h:
                    pickle.dump(inception_metrics_dict,h)
                    better_logger.info(f'saved FID and IS at {os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p")}' )


        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1






    #-----------------------------------------------------------------------------------

'''
    if write_log:
        utils.makedirs(args.save)
        logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

        logger.info(args)

        args_file_path = os.path.join(args.save, 'args.yaml')
        with open(args_file_path, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if args.distributed:
        if write_log: logger.info('Distributed initializing process group')
        torch.cuda.set_device(args.local_rank)
        distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=dist_utils.env_world_size(), rank=env_rank())
        assert(dist_utils.env_world_size() == distributed.get_world_size())
        if write_log: logger.info("Distributed: success (%d/%d)"%(args.local_rank, distributed.get_world_size()))

    # get deivce
    device = torch.device("cuda:%d"%torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)

    trainlog = os.path.join(args.save,'training.csv')
    testlog = os.path.join(args.save,'test.csv')

    traincolumns = ['itr','wall','itr_time','loss','bpd','fe','total_time','grad_norm'] ##lg
    testcolumns = ['wall','epoch','eval_time','bpd','fe', 'total_time', 'transport_cost']

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns).cuda() ##** what does the .cuda() do?
    if args.distributed: model = dist_utils.DDP(model,
                                                device_ids=[args.local_rank], 
                                                output_device=args.local_rank)

    traincolumns = append_regularization_keys_header(traincolumns, regularization_fns)

    if not args.resume and write_log:
        with open(trainlog,'w') as f:
            csvlogger = csv.DictWriter(f, traincolumns)
            csvlogger.writeheader()
        with open(testlog,'w') as f:
            csvlogger = csv.DictWriter(f, testcolumns)
            csvlogger.writeheader()

    set_cnf_options(args, model)

    if write_log: logger.info(model)
    if write_log: logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    if write_log: logger.info('Iters per train epoch: {}'.format(len(train_loader)))
    if write_log: logger.info('Iters per test: {}'.format(len(test_loader)))

    # optimizer
    if args.optimizer=='adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9,
                nesterov=False)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)


    # For visualization.
    if write_log: fixed_z = cvt(torch.randn(min(args.test_batch_size,100), *data_shape))

    if write_log:
        time_meter = utils.RunningAverageMeter(0.97)
        bpd_meter = utils.RunningAverageMeter(0.97)
        loss_meter = utils.RunningAverageMeter(0.97)
        steps_meter = utils.RunningAverageMeter(0.97)
        grad_meter = utils.RunningAverageMeter(0.97)
        tt_meter = utils.RunningAverageMeter(0.97)


    if not args.resume:
        best_loss = float("inf")
        itr = 0
        wall_clock = 0.
        begin_epoch = 1
    else:
        chkdir = os.path.dirname(args.resume)
        tedf = pd.read_csv(os.path.join(chkdir,'test.csv'))
        trdf = pd.read_csv(os.path.join(chkdir,'training.csv'))
        wall_clock = trdf['wall'].to_numpy()[-1]
        itr = trdf['itr'].to_numpy()[-1]
        best_loss = tedf['bpd'].min()
        begin_epoch = int(tedf['epoch'].to_numpy()[-1]+1) # not exactly correct

    if args.distributed:
        if write_log: logger.info('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())
    

    for epoch in range(begin_epoch, args.num_epochs + 1):
        if not args.validate:
            model.train()  # inheritated method from torch nn, activates 'train mode'

            with open(trainlog,'a') as f:
                if write_log: csvlogger = csv.DictWriter(f, traincolumns)

                for _, (x, y) in enumerate(train_loader):
                    start = time.time()
                    update_lr(optimizer, itr)
                    optimizer.zero_grad()

                    # cast data and move to device
                    x = add_noise(cvt(x), nbits=args.nbits)
                    #x = x.clamp_(min=0, max=1 )
                    
                    # compute loss
                    bpd, (x, z), reg_states = compute_bits_per_dim(x, model)
                    if np.isnan(bpd.data.item()):
                        raise ValueError('model returned nan during training')
                    elif np.isinf(bpd.data.item()):
                        raise ValueError('model returned inf during training')
                    
                    loss = bpd
                    if regularization_coeffs:
                        reg_loss = sum(
                            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                        )
                        loss = loss + reg_loss
                    total_time = count_total_time(model)

                    loss.backward()
                    nfe_opt = count_nfe(model)
                    if write_log: steps_meter.update(nfe_opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()


                    itr_time = time.time() - start
                    wall_clock += itr_time
                    
                    batch_size = x.size(0)
                    metrics = torch.tensor([1., batch_size,
                                            loss.item(),
                                            bpd.item(),
                                            nfe_opt,
                                            grad_norm,
                                            *reg_states]).float().cuda()

                    rv = tuple(torch.tensor(0.).cuda() for r in reg_states)

                    total_gpus, batch_total, r_loss, r_bpd, r_nfe, r_grad_norm, *rv = dist_utils.sum_tensor(metrics).cpu().numpy()


                    
                    if write_log:
                        time_meter.update(itr_time)
                        bpd_meter.update(r_bpd/total_gpus)
                        loss_meter.update(r_loss/total_gpus)
                        grad_meter.update(r_grad_norm/total_gpus)
                        tt_meter.update(total_time)

                        fmt = '{:.4f}'
                        logdict = {'itr':itr, 
                            'wall': fmt.format(wall_clock),
                            'itr_time': fmt.format(itr_time),
                            'loss': fmt.format(r_loss/total_gpus),
                            'bpd': fmt.format(r_bpd/total_gpus),
                            'total_time':fmt.format(total_time),
                            'fe': r_nfe/total_gpus,
                            'grad_norm': fmt.format(r_grad_norm/total_gpus),
                            }
                        if regularization_coeffs:
                            rv = tuple(v_/total_gpus for v_ in rv)
                            logdict = append_regularization_csv_dict(logdict,
                                    regularization_fns, rv)
                        csvlogger.writerow(logdict)

                        if itr % args.log_freq == 0:
                            log_message = (
                                    "Itr {:06d} | Wall {:.3e}({:.2f}) | "
                                    "Time/Itr {:.2f}({:.2f}) | BPD {:.2f}({:.2f}) | "
                                    "Loss {:.2f}({:.2f}) | "
                                    "FE {:.0f}({:.0f}) | Grad Norm {:.3e}({:.3e}) | "
                                    "TT {:.2f}({:.2f})".format(
                                    itr, wall_clock, wall_clock/(itr+1), 
                                    time_meter.val, time_meter.avg,
                                    bpd_meter.val, bpd_meter.avg,
                                    loss_meter.val, loss_meter.avg,
                                    steps_meter.val, steps_meter.avg,
                                    grad_meter.val, grad_meter.avg, 
                                    tt_meter.val, tt_meter.avg
                                    )
                                )
                            if regularization_coeffs:
                                log_message = append_regularization_to_log(log_message,
                                        regularization_fns, rv)
                            logger.info(log_message)



                    itr += 1

        # compute test loss
        model.eval()
        if args.local_rank==0:
            utils.makedirs(args.save)
            torch.save({
                "args": args,
                "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                "optim_state_dict": optimizer.state_dict(), 
                "fixed_z": fixed_z.cpu()
            }, os.path.join(args.save, "checkpt.pth"))
        if epoch % args.val_freq == 0 or args.validate:
            with open(testlog,'a') as f:
                if write_log: csvlogger = csv.DictWriter(f, testcolumns)
                with torch.no_grad():
                    start = time.time()
                    if write_log: logger.info("validating...")


                    lossmean = 0.
                    meandist = 0.
                    steps = 0
                    tt = 0.
                    for i, (x, y) in enumerate(test_loader):
                        sh = x.shape
                        x = shift(cvt(x), nbits=args.nbits)
                        loss, (x,z), _ = compute_bits_per_dim(x, model)
                        dist = (x.view(x.size(0),-1)-z).pow(2).mean(dim=-1).mean()
                        meandist = i/(i+1)*dist + meandist/(i+1)
                        lossmean = i/(i+1)*lossmean + loss/(i+1) 

                        tt = i/(i+1)*tt + count_total_time(model)/(i+1)
                        steps = i/(i+1)*steps + count_nfe(model)/(i+1)



                    loss = lossmean.item()
                    metrics = torch.tensor([1., loss, meandist, steps]).float().cuda()

                    total_gpus, r_bpd, r_mdist, r_steps = dist_utils.sum_tensor(metrics).cpu().numpy()
                    eval_time = time.time()-start

                    if write_log:
                        fmt = '{:.4f}'
                        logdict = {'epoch':epoch,
                                   'eval_time':fmt.format(eval_time),
                                   'bpd':fmt.format(r_bpd/total_gpus),
                                   'wall': fmt.format(wall_clock),
                                   'total_time':fmt.format(tt),
                                   'transport_cost':fmt.format(r_mdist/total_gpus),
                                   'fe':'{:.2f}'.format(r_steps/total_gpus)}

                        csvlogger.writerow(logdict)

                        logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}, Steps {:.4f}, TT {:.2f}, Transport Cost {:.2e}".format(epoch, eval_time, r_bpd/total_gpus, r_steps/total_gpus, tt, r_mdist/total_gpus))

                    loss = r_bpd/total_gpus


                    if loss < best_loss and args.local_rank==0: 
                        best_loss = loss
                        shutil.copyfile(os.path.join(args.save, "checkpt.pth"),
                                        os.path.join(args.save, "best.pth"))



            # visualize samples and density
            if write_log:
                with torch.no_grad():
                    fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
                    utils.makedirs(os.path.dirname(fig_filename))
                    generated_samples, _, _ = model(fixed_z, reverse=True)
                    generated_samples = generated_samples.view(-1, *data_shape)
                    nb = int(np.ceil(np.sqrt(float(fixed_z.size(0)))))
                    save_image(unshift(generated_samples, nbits=args.nbits), fig_filename, nrow=nb)
            if args.validate:
                break

'''

def main():
    parser = unet_utils.prepare_parser()
    config = vars(parser.parse_args())
    cudnn.benchmark = True ##**
    args = parser.parse_args()

    

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

    run(config,args)

    ## try:
    ##     with warnings.catch_warnings():
    ##         warnings.simplefilter("ignore", category=UserWarning)
    ##         main()
    ##     #if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    ## except Exception as e:
    ##     exc_type, exc_value, exc_traceback = sys.exc_info()
    ##     import traceback
    ##     traceback.print_tb(exc_traceback, file=sys.stdout)
    ##     # in case of exception, wait 2 hours before shutting down
    ##     #if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')



if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main()
        #if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        import traceback
        traceback.print_tb(exc_traceback, file=sys.stdout)
        traceback.print_exception(exc_type,exc_value,exc_traceback)