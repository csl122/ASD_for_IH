import argparse
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
import torchvision
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
# os.environ['RANK'] = '0'                     # 当前进程的rank设置为0
# os.environ['WORLD_SIZE'] = '2'               # 总共有4个进程参与训练
os.environ['MASTER_ADDR'] = 'localhost'      # master节点的地址，设置为本地IP地址或其它
os.environ['MASTER_PORT'] = '56780'           # 端口号设置，自己定

from ldm.data.dataset_halftone import HalftoneDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config, load_vae_from_config, norm


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))


def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        print('Auto_resume enabled.')
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state


def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsize",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/v1-5-pruned.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        default="models/vae-ft-mse-840000-ema-pruned.ckpt",
        help="path to checkpoint of vae model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/autoencoder_kl_32x32x4.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_vae",
        help="experiment name",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=832,
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--gpus",
        default=[0, 1], # [0, 1, 2, 3],
        help="gpu idx",
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )

    opt = parser.parse_args()
    return opt


def main(rank, world_size):
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    opt = parsr_args()
    config = OmegaConf.load(f"{opt.config}")

    print('=== 1 Distributed setting')
    # distributed setting
    init_dist(opt.launcher, rank=rank, world_size=world_size)
    
    print('=== 2 Torch setting')
    torch.backends.cudnn.benchmark = True
    device = rank
    torch.cuda.set_device(rank % 2)

    print('=== 3 Dataset loading')
    # dataset
    
    transforms = torchvision.transforms.Compose([
        # do *2 -1 to convert to [-1, 1] range
        torchvision.transforms.Lambda(norm)
    ])
    train_dataset = HalftoneDataset('datasets/val2017_256_FSDD', 'datasets/val2017_256', transform=transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    print('=== 4 AutoencoderKL model loading')
    # AutoencoderKL
    model = load_vae_from_config(config, vae_ckpt=f"{opt.vae_ckpt}").to(device)


    print('=== 5 To gpus')
    # to gpus
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank)

    print('=== 6 Optimiser setting')
    # optimizer
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['model']['base_learning_rate'])

    experiments_root = osp.join('experiments', opt.name)
    

    
    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        print(f'=== New experiment records in {experiments_root}')
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        print(f'=== Resuming experiment from {experiments_root}')
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    print('=== Training')
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        
        from tqdm import tqdm
        data_iter = tqdm(train_dataloader, desc=f"Train epoch {epoch}: ", disable=False)
        for _, data in enumerate(data_iter):
            current_iter += 1

            optimizer.zero_grad()
            
            loss = model.module.training_step(data, 0, 0)
            loss.backward()
            optimizer.step()
            
            data_iter.set_postfix(loss=loss.item())
            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(f'Epoch: {epoch}, Iteration: {current_iter + 1}')
                logger.info(f'Loss: {loss.item()}')

            # save checkpoint
            rank, _ = get_dist_info()
            if (rank == 0) and ((current_iter + 1) % config['training']['save_freq'] == 0):
                save_filename = f'model_vae_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                state_dict = model.state_dict()
                for key, param in state_dict.items():
                    if key.startswith('module.'):  # remove unnecessary 'module.'
                        key = key[7:]
                    save_dict[key] = param.cpu()
                torch.save(save_dict, save_path)
                # save state
                state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                save_filename = f'{current_iter + 1}.state'
                save_path = os.path.join(experiments_root, 'training_states', save_filename)
                torch.save(state, save_path)
                
    cleanup()

def cleanup():
    dist.destroy_process_group()

import torch.multiprocessing as mp
if __name__ == '__main__':
    
    world_size = torch.cuda.device_count()
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
