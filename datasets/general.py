from pathlib import Path
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import io
import imageio
from PIL import Image
import random
import torchvision.transforms as transforms

# Function to save the image patch
def save_image_patch(patch, filepath):
    patch = patch.permute(1, 2, 0).numpy() * 255
    patch_image = Image.fromarray(patch.astype(np.uint8))
    patch_image.save(filepath)


def jpegBlur(im,q):
    buf = io.BytesIO()
    imageio.imwrite(buf,im,format='jpg',quality=q)
    s = buf.getbuffer()
    return imageio.imread(s,format='jpg')



class GeneralDataset(Dataset):
    def __init__(self, root_dir, name, downgrade=None):
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.downgrade = None
        
        dirName = name
        if name == 'scene_completion': self.name_mask = True
        else: self.name_mask = False
        if name == 'in_the_wild': self.name_gt = False
        else: self.name_gt = True
        
        
        mask_dirName = dirName + "_GT"
        self.mask_dir = root_dir / mask_dirName
        
        if downgrade:
            if downgrade == 'facebook': dirName += "_Facebook"
            elif downgrade == 'whatsapp': dirName += "_Whatsapp"
            elif downgrade == 'wechat': dirName += "_Wechat"
            else: self.downgrade = downgrade
        
        img_dir = root_dir / dirName
        self.img_paths = list(img_dir.glob("*"))
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # Get image
        img = cv2.imread(str(img_path))[:, :, [2, 1, 0]]  # [H, W, C]
        assert img.dtype == np.uint8, "Image should be of type int!"
        assert (
            img.min() >= 0 and img.max() <= 255
        ), "Image should be bounded between [0, 255]!"
        
        
        if self.downgrade:
            if self.downgrade[0] == 'resize':
                img = cv2.resize(img, (0,0), fx=self.downgrade[1], fy=self.downgrade[1], interpolation=cv2.INTER_LINEAR)
            elif self.downgrade[0] == 'compression':
                img = jpegBlur(img, q=self.downgrade[1])
            else:
                assert True, "downgrade option is not available"
        
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        
        img_name = img_path.stem
        if self.name_mask:
            img_fullname = img_name + "_mask.png"
        elif self.name_gt:
            img_fullname = img_name + "_gt.png"
        else:
            img_fullname = img_name + ".png"
            
        map_path = self.mask_dir / img_fullname
        map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
        if self.downgrade:
            if self.downgrade[0] == 'resize':
                map = cv2.resize(map, (0,0), fx=self.downgrade[1], fy=self.downgrade[1], interpolation=cv2.INTER_LINEAR)
        assert map.dtype == np.uint8, "Ground-truth should be of type int!"
        assert (
            map.min() >= 0 and map.max() <= 255
        ), "Ground-truth should be bounded between [0, 255]!"

        map[map > 0] = 1
        
        
        map = torch.tensor(map)
        
        img_name = img_path.name

        return {"img": img, "label": 1, "map": map, "name": img_name, "path": img_path}

    def __len__(self):
        return len(self.img_paths)
    
    