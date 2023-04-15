from pathlib import Path

import cv2
import numpy as np
import toml
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as T

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
toPIL = T.ToPILImage()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        Resize((224,224), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _transform_2():
    return Compose([
        CenterCrop((224,224)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def preprocess(image: torch.Tensor, center_crop=False):
    image = toPIL(image)
    if center_crop: func = _transform_2()
    else: func =  _transform()
    return func(image)



class CASIA_1_Dataset(Dataset):
    def __init__(self, root_dir, spliced_only=False, center_crop=False):
        self.to_label = {"normal": 0, "splicing": 1}
        self.root_dir = Path(root_dir)
        self.center_crop = center_crop
        
        img_au_dir = self.root_dir / 'Au'
        self.au_img_paths = list(img_au_dir.glob("*"))
        
        img_tp_dir = self.root_dir / 'Tp' / 'Sp'
        self.tp_img_paths = list(img_tp_dir.glob("*"))
        
        self.map_dir = self.root_dir / 'CASIA 1.0 groundtruth' / 'Sp'
        
    def __len__(self):
        return len(self.au_img_paths) + len(self.tp_img_paths)
    
    def __getitem__(self, idx):
        """
        Returns
        -------
        Dict[str, Any]
            img : torch.ByteTensor
                [C, H, W], range [0, 255]
            label : int
                One of {0, 1}
            map : np.ndarray (uint8)
                [H, W], values one of {0, 1}
        """
        if idx < len(self.au_img_paths):
            img_path = self.au_img_paths[idx]
            label = 0
        else:
            img_path = self.tp_img_paths[idx-len(self.au_img_paths)]
            label = 1
        
        #Get image
        img = cv2.imread(str(img_path))[:,:,[2,1,0]]
        assert img.dtype == np.uint8, "Image should be of type int!"
        assert (
            img.min() >= 0 and img.max() <= 255
        ), "Image should be bounded between [0, 255]!"
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        _, height, width = img.shape
        img = preprocess(img, center_crop=self.center_crop)
        
        # # Get ground truth
        # if idx >= len(self.au_img_paths):
        #     img_prefex = img_path.name[:-4]
        #     map_path = self.map_dir / f"{img_prefex}_gt.png"
        #     map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
        #     assert map.dtype == np.uint8, "Ground-truth should be of type int!"
        #     assert (
        #         map.min() >= 0 and map.max() <= 255
        #     ), "Ground-truth should be bounded between [0, 255]!"
        #     # Resize map if doesn't match image
        #     if (height, width) != map.shape:
        #         map = cv2.resize(map, (width, height), interpolation=cv2.INTER_LINEAR)
        #     map[map > 0] = 1
        # else:
        #     map = np.zeros((height, width), dtype=np.uint8)
        # map = torch.tensor(map)
        # img_name = img_path.name
        return img, label         
    