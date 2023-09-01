# Instruction to run the project

## Prerequisites
Please use the `prereq.sh` script to install the required packages.

## Datasets
Please download required datasets and specify their paths in corresponding Python files.
> [datasets](https://drive.google.com/drive/folders/1X87ov9tv5zeB8JsfOeQLloRqORgnBJLQ?usp=sharing)

## Training
Please modify the `train_classifer.py` file for training the classifier. Two options are available: `--mode train` and `--mode test`

Please modify the `train_vae.py` file for training the VAE.

Please modify the `train_halftone.py` file for training the adapters and the prompt. For training arguments, please refer to the `useful.sh` script.

## Evaluation
Use `test_adapter.py` to do inference using list of halftone images in a txt file. Please refer to the `useful.sh` script for the arguments.

Use `valid_halftone.py` to get the PSNR and SSIM scores for a test dataset.

Use `valid_halftone_fid.py` to get the FID scores for a test dataset.


## Models
Models shall be stored in the `models` folder. Please refer to the `README.md` script in that folder for more information.
