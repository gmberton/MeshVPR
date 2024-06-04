
import os
import torch
import logging
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as tfm

import datasets.dataset_utils as dataset_utils

CUTOFF_LATITUDE = 37.785


def remove_art_prob(synt_path):
    """Synthetic images from SF have the filename equal to their real
    counterpart, with the only difference of having a field indicating the
    probability of artifacts existing (which is not used in MeshVPR. This
    field is removed to get the path of their real counterpart."""
    artifact_probability = synt_path.split("@")[-2]
    return synt_path.replace(artifact_probability, "")


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, real_dir, synt_dir, train_on_southern_half=False):
        super().__init__()
        self.real_dir = real_dir
        self.synt_dir = synt_dir
        self.synt_paths = dataset_utils.read_images_paths(synt_dir, get_abs_path=False)
        logging.info(f"All synthetic images: {len(self.synt_paths)}")
        
        if train_on_southern_half:
            # Keep only images with latitude < CUTOFF (lat is the 5th field within the filename separated by @)
            self.synt_paths = [p for p in tqdm(self.synt_paths) if float(p.split("@")[5]) < CUTOFF_LATITUDE]
            logging.info(f"Synthetic images in southern half of SF:  {len(self.synt_paths)}")
        
        self.transform = tfm.Compose([
            tfm.ToTensor(),
            tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def open_image(self, path):
        return self.transform(Image.open(path).convert("RGB"))
    
    def __getitem__(self, index):
        path = self.synt_paths[index]
        synt_path = os.path.join(self.synt_dir, path)
        real_path = os.path.join(self.real_dir, remove_art_prob(path))
        real_img = self.open_image(real_path)
        synt_img = self.open_image(synt_path)
        return real_img, synt_img, real_path, synt_path, index
    
    def __len__(self):
        return len(self.synt_paths)

