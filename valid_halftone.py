import os
import json
import cv2
import torch
from tqdm import tqdm
from basicsr.utils import tensor2img
from pytorch_lightning import seed_everything
from torch import autocast
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from ldm.util import contrast
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
    parser.add_argument(
        "--name",
        type=str,
        default="fsdd",
        help="experiment name",
    )
    opt = parser.parse_args()
    which_cond = opt.which_cond
    if opt.outdir is None:
        opt.outdir = f'outputs/valid-{opt.name}'
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
        image_paths = []
        for filenames in os.listdir(opt.prompt):
            if filenames.endswith(".png") or filenames.endswith(".jpg"):
                image_paths.append(os.path.join(opt.prompt, filenames))
        prompts = ["best quality, high quality, masterpiece, ultra high res, ultrarealistic, photorealistic, raw photo, detailed, 8k uhd, dslr, ultra-detailed"]*len(image_paths)
    print(image_paths[0])

    opt.vae_ckpt = "models/vae-ft-mse-840000-ema-pruned.ckpt"
    # prepare models
    sd_model, sampler = get_sd_models(opt)
    
    # getattr(ExtraCondition, which_cond) returns ID of the chosen condition
    # adapter model
    adapter = get_adapters(opt, getattr(ExtraCondition, which_cond))
    
    # require a model to convert an  ordinary image into specific condition type, e.g. depth map, pose.
    cond_model = None
    if opt.cond_inp_type == 'image':
        # condition type such as color does not require an additional model, return None
        cond_model = get_cond_model(opt, getattr(ExtraCondition, which_cond))

    # function to process the condition fig to certain size and ToTensor, the additional model will be applied here
    process_cond_module = getattr(api, f'get_cond_{which_cond}')

    opt.steps = 100
    opt.timesteps = None # set None for full steps
    
    
    # inference
    with torch.inference_mode(), \
            sd_model.ema_scope(), \
            autocast('cuda'):
        for test_idx, (cond_path, prompt) in enumerate(zip(image_paths, prompts)):
            if os.path.exists(os.path.join(opt.outdir, cond_path[-16:])):
                continue
            seed_everything(opt.seed)
            for v_idx in range(opt.n_samples):
                # seed_everything(opt.seed+v_idx+test_idx)
                # get the condition as a tensor for feeding into a NN
                cond = process_cond_module(opt, cond_path, opt.cond_inp_type, cond_model)

                z = sd_model.encode_first_stage((cond * 2 - 1.))
                z = sd_model.get_first_stage_encoding(z)
                # x_T = sd_model.q_sample(z, sampler.ddim_timesteps[opt.timesteps-1]) # 也可能减2
                
                # save condition image
                adapter_features, append_to_context = get_adapter_feature(cond, adapter)
                opt.prompt = prompt
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context, z=z, timesteps=opt.timesteps)
                cv2.imwrite(os.path.join(opt.outdir, cond_path[-16:]), tensor2img(result))
    
    log = open(os.path.join(opt.outdir, '_udd.txt'), 'w')
    psnr_ssim('datasets/val2017_256', opt.outdir, log)
    contrast(opt.outdir)
    
def psnr_ssim(test_target_path, test_output_path, log, color=False):
    state = {
        'IMG': '',
        'psnr': 0.0,
        'ssim': 0.0,
    }

    avg_psnr = 0
    avg_ssim = 0
    count = 0
    
    test_tqdm = tqdm(os.listdir(test_output_path))
    for filename in test_tqdm:
    # for filename in os.listdir(test_output_path):
        if filename.startswith('_'):
            continue
        # print('======> psnr-ssim ', filename)

        img1_path = os.path.join(test_output_path, filename)
        img2_path = os.path.join(test_target_path, filename)

        img1 = cv2.imread(img1_path) if color else cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        # resize img1 to img2
        # img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        img2 = cv2.imread(img2_path) if color else cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        psnr = compare_psnr(img1, img2)
        ssim = compare_ssim(img1, img2, channel_axis=2 if color else None)

        state['IMG'] = filename
        # state['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        state['psnr'] = psnr
        state['ssim'] = ssim
        avg_psnr += psnr
        avg_ssim += ssim
        count += 1
        log.write('%s\n' % json.dumps(state))
        log.flush()

    print('======> total img count ', count)
    log.write('%s\n' % json.dumps(
        {'avg_psnr': avg_psnr / count, 'avg_ssim': avg_ssim / count}))
    log.flush()
    log.close()
    print('done!!!')
    

if __name__ == '__main__':
    main()
    # show the image in frequency domain using fft
    # import cv2
    # import numpy as np
    # image = cv2.imread('outputs/validation/valid-fsdd_3_lprompt/000000000724.jpg')[:,:,0]
    # #apply fft
    # image_fft = np.fft.fft2(image)
    # # shift the fft output so that low frequencies are in the center
    # image_fft_shift = np.fft.fftshift(image_fft)
    # # calculate the magnitude
    # magnitude_spectrum = 20*np.log(np.abs(image_fft_shift))
    # print(np.min(magnitude_spectrum), np.max(magnitude_spectrum))
    # # normalize the magnitude spectrum
    # magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # print(np.min(magnitude_spectrum), np.max(magnitude_spectrum))
    # # save the image
    # cv2.imwrite('_fft.jpg', magnitude_spectrum)
    
# python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone/models/model_ad_25000.pth

