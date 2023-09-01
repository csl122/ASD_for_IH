from ldm.data.dataset_halftone import HalftoneDataset
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

fid = FrechetInceptionDistance(feature=2048, normalize=True)

halftone_path = 'outputs/test'
original_path = 'datasets/val2017_256'
valid_dataset = HalftoneDataset(halftone_path, original_path)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

fids = []

for _, data in enumerate(tqdm(valid_dataloader, desc=f"Computing FID score: ", disable=False)):
    original = data['original']
    halftone = data['halftone']
    
    
    fid.update(original, real=True)
    fid.update(halftone, real=False)
    
    score = fid.compute()
    print(score)
    
    fids.append(score)
    
avg = sum(fids) / len(fids)
print(f"Average FID score: {avg}, {score}")

