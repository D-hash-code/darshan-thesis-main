#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license)
# some modifications in class Generator and G_D
# new class "Unet_Discriminator" based on original class "Discriminator"
import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

import u_net.layers as layers
import lib.odenvp as odenvp

import copy
from matplotlib import pyplot as plt


import lib.layers as ODElayers
from lib.layers.odefunc import ODEnet
from lib.layers.squeeze import squeeze, unsqueeze
import numpy as np

from train_misc import create_regularization_fns


from torch.optim.optimizer import Optimizer
class Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(Adam16, self).__init__(params, defaults)

  # Safety modification to make sure we floatify our state
  def load_state_dict(self, state_dict):
    super(Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        grad = p.grad.data.float()
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          # Fp32 copy of the weights
          state['fp32_p'] = p.data.float()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].half()

    return loss


def D_unet_arch(ch=64, attention='0',ksize='333333', dilation='111111',out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [3] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}


    arch[256] = {'in_channels' :            [3] + [ch*item for item in [1, 2, 4, 8, 8, 16, 8*2, 8*2, 4*2, 2*2, 1*2  , 1         ]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 8,   8,   4,   2,   1,   1          ]],
                             'downsample' : [True] *6 + [False]*6 ,
                             'upsample':    [False]*6 + [True] *6,
                             'resolution' : [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256 ],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,13)}}



    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=256,
                             D_kernel_size=3, D_attn='0',
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                             SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
                             D_init='ortho', skip_init=False, D_param='SN', decoder_skip_connection = True, **kwargs):
        super(Unet_Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16



        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier  )[resolution]

        self.unconditional = True

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)

            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                            eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         #which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition: #index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                                                         self.which_conv)]


        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])


        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)
        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        self.linear_middle = self.which_linear(16*self.ch, output_dim)

        # Initialize weights
        if not skip_init:
            self.init_weights()

        ###
        print("_____params______")
        for name, param in self.named_parameters():
            print(name, param.size())

        # Set up optimizer
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            print('Using fp16 adam in D...')
            self.optim = Adam16(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                                         betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        # LR scheduling, left here for forward compatibility
        # self.lr_sched = {'itr' : 0}# if self.progressive else {}
        # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)



    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0


        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0

        out = out + proj

        out = out.view(out.size(0),1,self.resolution,self.resolution)

        return out, bottleneck_out



class Generator(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(self, args, config, G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8):
        super(Generator, self).__init__()

        self.fp16 = False
        hidden_dims = tuple(map(int, args.dims.split(",")))
        strides = tuple(map(int, args.strides.split(",")))

        regularization_fns, regularization_coeffs = create_regularization_fns(args)
        
        data_shape=(3,128,128)
        input_size = (args.batch_size, *data_shape)
        squeeze_first=args.squeeze_first


        if squeeze_first:
            bsz, c, w, h = input_size
            c, w, h = c*4, w//2, h//2
            input_size = bsz, c, w, h
        self.n_scale = self._calc_n_scale(input_size) #min(args.n_scale, self._calc_n_scale(input_size))
        self.n_blocks = args.num_blocks
        self.intermediate_dims = hidden_dims
        self.layer_type=args.layer_type
        self.zero_last=args.zero_last
        self.div_samples=args.div_samples
        self.nonlinearity = args.nonlinearity
        self.strides=strides
        self.squash_input = True
        self.alpha = args.alpha
        self.squeeze_first = args.squeeze_first
        self.cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns}

        self.dim_z = 128
        self.resolution = 128
        self.unconditional = True

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        #self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps

        self.optim = optim.Adam(params=self.parameters(), lr=config['G_lr'],
                                                    betas=(config['G_B1'], config['G_B2']), weight_decay=0,
                                                    eps=config['adam_eps'])

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    div_samples=self.div_samples,
                    zero_last=self.zero_last,
                    layer_type=self.layer_type,
                    strides=self.strides,
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=(ODElayers.LogitTransform(self.alpha) if self.alpha > 0 else ODElayers.ZeroMeanTransform())
                    if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)


    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, y=None, logpx=None, reg_states=tuple(), reverse=True, density=False):
        if reverse:
            out = self._generate(x, logpx, reg_states,density=density)
            if self.squeeze_first:
                x = unsqueeze(out[0])
            else:
                x = out[0]
            return x, out[1], out[2]
        else:
            if self.squeeze_first:
                x = squeeze(x)
            return self._logdensity(x, logpx, reg_states)

    def _logdensity(self, x, logpx=None, reg_states=tuple()):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx, reg_states = self.transforms[idx].forward(x, _logpx, reg_states)
            if idx < len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        return out, _logpx, reg_states

    def _generate(self, z, logpz=None, reg_states=tuple(),density=False):
        z = z.view(z.shape[0], -1)
        zs = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims) #256x256
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)] # I believe this is squeezing the noise/latent to match the output of the 'final' layer?
        _logpz = logpz
        z_prev, _logpz, _ = self.transforms[-1](zs[-1], _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1): # if len(self.transforms) is 10 then idx will be 8,7,..,1,0
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz, reg_states = self.transforms[idx](z_prev, _logpz, reg_states, reverse=True)
        return z_prev, _logpz, reg_states


class StackedCNFLayers(ODElayers.SequentialFlow):
    def __init__(
        self,
        initial_size,
        idims=(32,),
        nonlinearity="softplus",
        layer_type="concat",
        div_samples=1,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        zero_last=True,
        strides=None,
        cnf_kwargs={},
    ):
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = ODEnet(idims, size, strides, True, layer_type=layer_type, nonlinearity=nonlinearity, zero_last_weight=zero_last)
            f = ODElayers.ODEfunc(net, div_samples=div_samples)
            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [ODElayers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]
            post = [ODElayers.CNF(_make_odefunc(after_squeeze_size), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [ODElayers.SqueezeLayer(2)] + post
        else:
            chain += [ODElayers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]

        super(StackedCNFLayers, self).__init__(chain)


#-----------------------------------------------------
        

# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D, config):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

        self.config = config

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
                            split_D=False, dw1=[],dw2=[], reference_x = None, mixup = False, mixup_only = False, target_map=None):

        print('z_[:batchsize] shape: ', z.shape)
        if mixup:
            gy = dy
            #why? so the mixup samples consist of same class

        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):

            G_z = self.G(z, gy)
            # Cast as necessary
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()

        if mixup:
            initial_x_size = x.size(0)

            mixed = target_map*x+(1-target_map)*G_z
            mixed_y = dy


        if not mixup_only:
            # we get here in the cutmix cons extra case
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            dmap = torch.tensor([])
            if mixup:
                #we get here in the cutmix  "consistency loss and augmentation" case, if "mixup" is true for the current round (depends on p mixup)
                D_input = torch.cat([D_input, mixed], 0)
                if self.config["dataset"]!="coco_animals":
                    D_class = torch.cat([D_class.float(), mixed_y.float()], 0)
                else:
                    D_class = torch.cat([D_class.long(), mixed_y.long()], 0)
        else:
            #not reached in cutmix "consistency loss and augmentation"
            D_input = mixed
            D_class = mixed_y
            dmap = torch.tensor([])

            del G_z
            del x
            G_z = None
            x = None

        D_out, D_middle = self.D(D_input, D_class)

        del D_input
        del D_class


        if x is not None:

            if not mixup:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0]])     # D_fake, D_real
            else:
                out = torch.split(D_out, [G_z.shape[0], x.shape[0], mixed.shape[0]])  # D_fake, D_real, D_mixed
            out = out + (G_z,)
            if mixup:
                out = out + (mixed,)

            if not mixup:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0]])     # D_middle_fake, D_middle_real
            else:
                D_middle =  torch.split(D_middle, [G_z.shape[0], x.shape[0] , mixed.shape[0]])
            out = out + D_middle
            ###return target map as well
            if mixup:
                out = out + (target_map,)

            return out


        else:
            #in mixup# you arrive here
            out = (D_out,)

            if return_G_z:
                out = out + (G_z,)
            if mixup_only:
                out = out + (mixed,)

            out =  out + (D_middle,)
            ##return target map as well
            if mixup:
                out = out + (target_map,)

            return out
