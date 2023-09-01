
# test original sd
python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_v1/models/model_ad_18000.pth

# test trained vae
python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_v1_lprompt_bettervae/models/model_ad_26000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt

# test trained vae on fsdd
python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist_fsdd_3.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_FSDD_adapter_lprompt/models/model_ad_26000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt --learn_prompt

# test trained vae on bdd
python test_adapter.py --which_cond halftone --cond_inp_type image --prompt testlist_bdd.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_BDD_adapter_lprompt/models/model_ad_26000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt

# Train adapter part from scratch
python train_halftone.py --name train_halftone_adapter_FSDD_adapter --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt

# Train adapter part from history
python train_halftone.py --auto_resume --adapter_ckpt experiments/train_halftone_adapter_BDD_adapter/models/model_ad_12000.pth --name train_halftone_adapter_BDD_adapter --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt

# Train adapter part from scratch, with a learned adapter
python train_halftone.py --learn_prompt --adapter_ckpt experiments/train_halftone_adapter_FSDD_adapter/models/model_ad_26000.pth --name train_halftone_adapter_FSDD_adapter_lprompt --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt

# Train VAE from a pretrained model defined in config
python train_vae.py

python test_vaee.py --which_cond halftone --cond_inp_type image --prompt testlist.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_v1/models/model_ad_18000.pth 

python train_halftone.py --adapter_ckpt experiments/train_halftone_adapter_v1/models/model_ad_18000.pth --name train_halftone_adapter_v1_test_bettervae --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt


# validation using a list in a txt
python valid_halftone.py --which_cond halftone --cond_inp_type image --prompt testlist_bdd.txt --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_BDD_adapter_lprompt/models/model_ad_26000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt --name bdd_lprompt --learn_prompt

# validation using a folder
python valid_halftone.py --which_cond halftone --cond_inp_type image --prompt datasets/val2017_256_UDD --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_UDD_adapter_lprompt/models/model_ad_10000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt --name udd_lprompt_none --learn_prompt

# validation non-prompt-learning using a folder
python valid_halftone.py --which_cond halftone --cond_inp_type image --prompt datasets/val2017_256_FSDD_3 --sd_ckpt models/v1-5-pruned.ckpt --resize_short_edge 256 --cond_tau 1.0 --cond_weight 1.0 --n_samples 1 --adapter_ckpt experiments/train_halftone_adapter_FSDD_adapter/models/model_ad_26000.pth --vae_ckpt models/vae-ft-mse-840000-ema-pruned.ckpt --name fsdd_none
