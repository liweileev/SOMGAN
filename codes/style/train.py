'''
Author: Liweileev
Date: 2022-01-03 17:00:02
LastEditors: Liweileev
LastEditTime: 2022-01-31 01:10:10
'''

import click
import dnnlib
import os
import re
import json
import torch
import tempfile

from training import training_loop
from torch_utils import training_stats
from torch_utils import custom_ops

class UserError(Exception):
    pass

def setup_training_loop_kwargs(
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    # Dataset.
    data       = None, # Training dataset (required): <path>
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False
    # Base config.
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>
    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    num_d         = None, # Number of discriminators: <int>, default = 1
    topo       = None, # Topology constraint type: 'grid' (default), 'triangle'
    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), <file>, <url>
):

    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics
    # ------------------------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap
    
    args.metrics = ['fid50k_full']

    # -----------------------------------
    # Dataset: data, mirror
    # -----------------------------------
    
    assert data is not None
    assert isinstance(data, str)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, xflip=False)
    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        desc = training_set.name
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')
    
    if num_d is None:
        num_d = 1
    assert isinstance(num_d, int)
    args.num_D = num_d
    desc += f'-{num_d:d}D'

    assert topo is None or isinstance(topo, str)
    if topo is None:
        desc += '-grid'
        args.topo = 'grid'
    else:
        desc += '-' + topo
        args.topo = topo # custom path or url

    if mirror is None:
        mirror = False
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True
    
    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    res = args.training_set_kwargs.resolution
    fmaps = 1 if res >= 512 else 0.5
    lrate = 0.002 if res >= 1024 else 0.0025
    default_map = 2
    default_kimg = 25000
    default_mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
    default_mbstd = min(default_mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
    default_gamma = 0.0002 * (res ** 2) / default_mb # heuristic formula
    default_ema = default_mb * 10 / 32

    args.G_kwargs = dnnlib.EasyDict(class_name='training.Generators.StyleGenerator.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.Discriminators.StyleDiscriminator.Discriminator', epilogue_kwargs=dnnlib.EasyDict())
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.G_kwargs.mapping_kwargs.num_layers = default_map
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = default_mbstd

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.SOMGANLoss', r1_gamma=default_gamma)

    args.total_kimg = default_kimg
    args.batch_size = default_mb
    args.batch_gpu = default_mb // gpus
    args.ema_kimg = default_ema
    args.ema_rampup = None

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6
        bgc = dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **bgc)
        desc += f'-ada'
    elif aug == 'noaug':
        pass
    else:
        raise UserError(f'--aug={aug} not supported')
    
    # ----------------------------------
    # Transfer learning: resume
    # ----------------------------------

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup
    
    desc += f'-{gpus:d}gpu'

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)
    
    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'
    # Execute training loop.
    training_loop.training_loop(rank=rank, **args)

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Base config.
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# Discriminator augmentation.
@click.option('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']))
@click.option('--num_d', help='Number of discriminators [default: 1', type=int, metavar='INT')
@click.option('--topo', help='Topology constraint type [default: grid]', type=click.Choice(['grid', 'triangle']))

# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')

def main(ctx, outdir, dry_run, **config_kwargs):
    dnnlib.util.Logger(should_flush=True)
    
    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:                               {args.run_dir}')
    print(f'Training data:                                  {args.training_set_kwargs.path}')
    print(f'Training duration:                              {args.total_kimg} kimg')
    print(f'Number of GPUs:                                 {args.num_gpus}')
    print(f'Number of Discriminators:                       {args.num_D}')
    print(f'Topological constraints of Discriminators:      {args.topo}')
    print(f'Image resolution:                               {args.training_set_kwargs.resolution}')
    print(f'Dataset x-flips:                                {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)
    
    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)
    
if __name__ == "__main__":
    main()