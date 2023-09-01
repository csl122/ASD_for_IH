import argparse
import logging
import os
import os.path as osp
import torch
import torch.distributed as dist
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir)
from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
# os.environ['RANK'] = '0'                     # 当前进程的rank设置为0
# os.environ['WORLD_SIZE'] = '2'               # 总共有4个进程参与训练
os.environ['MASTER_ADDR'] = 'localhost'      # master节点的地址，设置为本地IP地址或其它
os.environ['MASTER_PORT'] = '56780'           # 端口号设置，自己定
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'           # 端口号设置，自己定

from ldm.data.dataset_halftone import HalftoneDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config


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
        default=32,
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
        default=None,
        help="path to checkpoint of vae model",
    )
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default=None,
        help="path to checkpoint of adapter model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/sd-v1-train.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_halftone_adapter",
        help="experiment name",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=78,
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
    parser.add_argument(
        "--learn_prompt",
        action='store_true',
        help="train prompt embedding or not",
    )
    parser.add_argument(
        "--prompt_lr",
        default=1e-4, 
        type=float,
        help="learning rate for training prompt embedding",
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
    halftone_path = 'datasets/train2017_256_FSDD'
    original_path = 'datasets/train2017_256'
    train_dataset = HalftoneDataset(halftone_path, original_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=(train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    print('=== 4 Stable diffusion model loading')
    # stable diffusion
    model = load_model_from_config(config, f"{opt.ckpt}", vae_ckpt=opt.vae_ckpt).to(device)


    print('=== 5 Halftone encoder model loading')
    # halftone encoder
    model_ad = Adapter(cin=int(3 * opt.f * opt.f), channels=[320, 640, 1280, 1280], nums_rb=2, ksize=1, sk=True, use_conv=False).to(
        device)
    if opt.adapter_ckpt is not None:
        print(f'=== 5.1 Training adapter from loaded ckpt: {opt.adapter_ckpt}')
        adapter_ckpt = torch.load(opt.adapter_ckpt, map_location='cpu')
        m, u = model_ad.load_state_dict(adapter_ckpt, strict=False)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)
        model.to(device)
    else:
        print('=== 5.1 Training adapter from zero')


    print('=== 7 Optimiser setting')
    # optimizer
    base_lr = config['training']['lr']
    my_list = ['module.cond', 'cond']
    params = list(filter(lambda kv: kv[0] in my_list, model_ad.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model_ad.named_parameters()))
    params = [i[1] for i in params]
    base_params = [i[1] for i in base_params]
    # freeze base params if learn_prompt is True
    if opt.learn_prompt:
        print('=== 7.1 Freeze adapter params, only train prompt embedding')
        base_lr = 0.
        for param in base_params:
            param.requires_grad = False
    else:
        print('=== 7.1 Train only adapter params')
        opt.prompt_lr = 0.
        base_lr = config['training']['lr']
        for param in params:
            param.requires_grad = False
    optimizer = torch.optim.AdamW([
                            {'params': list(filter(lambda p: p.requires_grad, base_params))},
                            {'params': list(filter(lambda p: p.requires_grad, params)),
                                        'lr': opt.prompt_lr, 'weight_decay': 1e-3}
                        ], lr=base_lr)
    print(f"=== 7.2 param_groups: {optimizer.state_dict()['param_groups']}")
    # params = list(model_ad.parameters())
    # optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])
    print(f'=== 7.3 Learning rate: [Prompt: {opt.prompt_lr}, Base: {base_lr}]\n=== 7.4 Learning parameters{[name for name, param in model_ad.named_parameters() if param.requires_grad]}')

    print('=== 6 To gpus')
    # to gpus
    model_ad = torch.nn.parallel.DistributedDataParallel(
        model_ad,
        device_ids=[rank],
        output_device=rank,)
        # find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank)


    experiments_root = osp.join('experiments', opt.name)
    

    
    # resume state
    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        print(f'=== 8 New experiment records in {experiments_root}')
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        print(f'=== 8 Resuming experiment from {experiments_root}')
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

    prompt = "best quality, high quality, masterpiece, ultra high res, ultrarealistic, photorealistic, raw photo, detailed, 8k uhd, dslr, ultra-detailed"
    with torch.no_grad():
        init_cond = model.module.get_learned_conditioning([prompt])
    model_ad.module.set_conditioning(init_cond.data)
    logger.info(f'=== Training using {halftone_path}')
    print('=== 9 Training')
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    for epoch in range(start_epoch, opt.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        # train
        
        from tqdm import tqdm
        for _, data in enumerate(tqdm(train_dataloader, desc=f"Train epoch {epoch}: ", disable=False)):
            current_iter += 1
            with torch.no_grad():
                c = model.module.get_learned_conditioning(data['sentence'])
                z = model.module.encode_first_stage((data['original'] * 2 - 1.).to(device))
                z = model.module.get_first_stage_encoding(z)

            optimizer.zero_grad()
            model.zero_grad()
            
            # get halftone features and learned prompt as cond
            features_adapter, cond = model_ad(data['halftone'].to(device))
            prompt_cond = cond if opt.learn_prompt else c
                        
            assert prompt_cond.shape == c.shape
            
            l_pixel, loss_dict = model(z, c=prompt_cond, features_adapter=features_adapter)
            l_pixel.backward()
            optimizer.step()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(f'Epoch: {epoch}, Iteration: {current_iter + 1}')
                logger.info(loss_dict)

            # save checkpoint
            rank, _ = get_dist_info()
            if (rank == 0) and ((current_iter + 1) % config['training']['save_freq'] == 0):
                save_filename = f'model_ad_{current_iter + 1}.pth'
                save_path = os.path.join(experiments_root, 'models', save_filename)
                save_dict = {}
                state_dict = model_ad.state_dict()
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
