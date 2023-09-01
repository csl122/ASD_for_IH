import argparse
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from basicsr.utils import tensor2img
from omegaconf import OmegaConf
# os.environ['RANK'] = '0'                     # 当前进程的rank设置为0
# os.environ['WORLD_SIZE'] = '2'               # 总共有4个进程参与训练
os.environ['MASTER_ADDR'] = 'localhost'      # master节点的地址，设置为本地IP地址或其它
os.environ['MASTER_PORT'] = '56780'           # 端口号设置，自己定

from ldm.data.dataset_halftone import HalftoneDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config, load_vae_from_config
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

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
        default=1,
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
        # default="experiments/train_vae/models/model_vae_125000.pth",
        # default="models/diffusion_pytorch_model.bin",
        help="path to checkpoint of vae model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/autoencoder/autoencoder_kl_test.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="test_vae",
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
    train_dataset = HalftoneDataset('datasets/val2017_256_FSDD', 'datasets/val2017_256')
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
    model.eval()

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
    
    mkdir_and_rename(experiments_root)

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    print('=== Testing')
    # Testing
    current_iter = 0
    with torch.no_grad():
        from tqdm import tqdm
        for _, data in enumerate(tqdm(train_dataloader, desc=f"Testing: ", disable=False)):
            current_iter += 1
            
            
            # z = model.module.encode((data['original'] * 2 - 1.).to(device)).sample()
            recon, post = model((data['original'] * 2 - 1.).to(device))
            # recon = model.module.decode(z)
            x_samples = torch.clamp((recon + 1.0) / 2.0, min=0.0, max=1.0)
            # save reconstruction images to experiment root 
            # import cv2
            # cv2.imwrite(osp.join(experiments_root, f"recon_{current_iter}.png"), tensor2img(x_samples))
            import torchvision
            torchvision.utils.save_image(x_samples, osp.join(experiments_root, f"recon_{current_iter}.png"))
                
            
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
