import os

import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)
from ldm.util import load_model_from_config, load_vae_from_config


torch.set_grad_enabled(False)


def main():
    supported_cond = [e.name for e in ExtraCondition]
    parser = get_base_argument_parser()
    parser.add_argument(
        '--which_cond',
        type=str,
        required=True,
        choices=supported_cond,
        help='which condition modality you want to test',
    )
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/test-{which_cond}'
    os.makedirs(opt.outdir, exist_ok=True)
    if opt.resize_short_edge is None:
        print(f"you don't specify the resize_shot_edge, so the maximum resolution is set to {opt.max_resolution}")
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # support two test mode: single image test, and batch test (through a txt file)
    if opt.prompt.endswith('.txt'):
        assert opt.prompt.endswith('.txt')
        image_paths = []
        prompts = []
        with open(opt.prompt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_paths.append(line.split('; ')[0])
                # prompts.append(line.split('; ')[1])
                prompts.append('')
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]
    print(image_paths)

    # opt.vae_ckpt = "experiments/train_vae/models/model_vae_125000.pth"
    opt.vae_ckpt = "models/vae-ft-mse-840000-ema-pruned.ckpt"
    # prepare models
    sd_model, sampler = get_sd_models(opt)
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/autoencoder/autoencoder_kl_test.yaml")
    model = load_vae_from_config(config, vae_ckpt=f"{opt.vae_ckpt}").to(opt.device)
    
    # getattr(ExtraCondition, which_cond) returns ID of the chosen condition
    # adapter model
    # adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    
    # require a model to convert an  ordinary image into specific condition type, e.g. depth map, pose.
    cond_model = None
    if opt.cond_inp_type == 'image':
        # condition type such as color does not require an additional model, return None
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    # function to process the condition fig to certain size and ToTensor, the additional model will be applied here
    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    opt.steps = 200
    opt.timesteps = 150
    
    
    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                # get the condition as a tensor for feeding into a NN
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)
                # print(cond.shape)
                # print(cond)
                z = sd_model.encode_first_stage((cond * 2 - 1.)).sample()
                # z = sd_model.get_first_stage_encoding(z)
                # x_T = sd_model.q_sample(z, sampler.ddim_timesteps[opt.timesteps-1]) # 也可能减2
                
                # save condition image
                base_count = len(os.listdir(opt.outdir)) // 2
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))

                # adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                # result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context, z=z, timesteps=opt.timesteps)
                # x_samples = sd_model.first_stage_model.decode(z)
                # x_samples, post = sd_model.first_stage_model((cond * 2 - 1.))
                x_samples, post = model((cond * 2 - 1.).to(opt.device))
                result = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


if __name__ == '__main__':
    main()
# python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone/models/model_ad_25000.pth

