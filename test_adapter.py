import os

import cv2
import torch
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.inference_base import (diffusion_inference, get_adapters, get_base_argument_parser, get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_model)

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
                prompts.append("best quality, high quality, masterpiece, ultra high res, ultrarealistic, photorealistic, raw photo, detailed, 8k uhd, dslr, ultra-detailed")
    else:
        image_paths = [opt.cond_path]
        prompts = [opt.prompt]
    print(image_paths)

    opt.vae_ckpt = "models/vae-ft-mse-840000-ema-pruned.ckpt"
    # prepare models
    sd_model, sampler = get_sd_models(opt)
    
    # getattr(ExtraCondition, which_cond) returns ID of the chosen condition
    # adapter model
    # TODO: change the path to the adapter ckpt
    opt.adapter_ckpt = 'experiments/train_halftone_adapter_FSDD_adapter_lprompt/models/model_ad_26000.pth'
    adapter_fsdd = get_adapters(opt, getattr(ExtraCondition, which_cond))
    opt.adapter_ckpt = 'experiments/train_halftone_adapter_BDD_adapter_lprompt/models/model_ad_26000.pth'
    adapter_bdd = get_adapters(opt, getattr(ExtraCondition, which_cond))
    opt.adapter_ckpt = 'experiments/train_halftone_adapter_UDD_adapter_lprompt/models/model_ad_26000.pth'
    adapter_udd = get_adapters(opt, getattr(ExtraCondition, which_cond))
    opt.adapter_ckpt = 'experiments/train_halftone_adapter_KDD_adapter_lprompt/models/model_ad_26000.pth'
    adapter_kdd = get_adapters(opt, getattr(ExtraCondition, which_cond))
    
    # TODO: change the path to the classifier ckpt
    from train_classifier import ResNet
    classifier = ResNet(num_classes=4).cuda()
    state_dict = torch.load('experiments/classifier/best_model.pt')
    classifier.load_state_dict(state_dict, strict=False)
    
    # require a model to convert an  ordinary image into specific condition type, e.g. depth map, pose.
    cond_model = None
    if opt.cond_inp_type == 'image':
        # condition type such as color does not require an additional model, return None
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    # function to process the condition fig to certain size and ToTensor, the additional model will be applied here
    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    opt.steps = 200
    opt.timesteps = 195 # set None for full steps
    
    
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
                classifier.eval()
                label = classifier(cond).argmax(dim=1)[0]
                print(f"Halftone type: {label}")
                if label == 0:
                    adapter = adapter_fsdd
                elif label == 1:
                    adapter = adapter_bdd
                elif label == 2:
                    adapter = adapter_udd
                elif label == 3:
                    adapter = adapter_kdd

                z = sd_model.encode_first_stage((cond * 2 - 1.))
                z = sd_model.get_first_stage_encoding(z)
                # x_T = sd_model.q_sample(z, sampler.ddim_timesteps[opt.timesteps-1]) # 也可能减2
                
                # save condition image
                base_count = len(os.listdir(opt.outdir)) // 2
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_{which_cond}.png'), tensor2img(cond))

                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context, z=z, timesteps=opt.timesteps)
                cv2.imwrite(os.path.join(opt.outdir, f'{base_count:05}_result.png'), tensor2img(result))


if __name__ == '__main__':
    main()
# python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone/models/model_ad_25000.pth

