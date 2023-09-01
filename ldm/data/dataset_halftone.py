import json
import cv2
import os
from basicsr.utils import img2tensor


class HalftoneDataset():
    def __init__(self, halftone_path, original_path, transform=None):
        super(HalftoneDataset, self).__init__()

        self.halftone_path = halftone_path
        self.original_path = original_path
        self.image_name_list = os.listdir(halftone_path)
        self.transform = transform


    def __getitem__(self, idx):
        halftone_name = self.image_name_list[idx]

        halftone = cv2.imread(os.path.join(self.halftone_path, halftone_name))
        halftone = img2tensor(halftone, bgr2rgb=True, float32=True) / 255.

        original = cv2.imread(os.path.join(self.original_path, halftone_name))  # [:,:,0]
        original = img2tensor(original, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.
        
        if self.transform is not None:
            halftone = self.transform(halftone)
            original = self.transform(original)

        prompt = "best quality, high quality, masterpiece, ultra high res, ultrarealistic, photorealistic, raw photo, detailed, 8k uhd, dslr, ultra-detailed"
        # sentence = ""
        sentence = prompt

        return {'original': original, 'halftone': halftone, 'sentence': sentence}

    def __len__(self):
        return len(self.image_name_list)
