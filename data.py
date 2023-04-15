import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
import torch
import random
import json
import glob


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(res):
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        RandomCrop((res,res)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def preprocess(image: torch.Tensor, res):
    func =  _transform(res)
    return func(image)


class ExampleDataset(Dataset):
    def __init__(self, root_path, resolution=224):
        super().__init__()
        self.image_paths = glob.glob(f"{root_path}/*.png")
        self.resolution = resolution
        print("dataset length: ", len(self))
    
    def __len__(self):
        return len(self.image_paths)
    
    
    def get_possible_item(self, index):
        imagepath = self.image_paths[index]
        image = Image.open(imagepath)
        image = preprocess(image, self.resolution)

        with open(imagepath.replace('png', 'txt')) as f:
            exif_str = json.load(f)['exif']
            
        example = {
            "imgpath": imagepath, 
            "image": image, 
            "exif": exif_str,
        }
        return example 
    
    def __getitem__(self, index):
        success = False
        while not success:
            try:
                example = self.get_possible_item(index)
                success = True
            except Exception as e:
                print("!!!error ", index, e)
                index = random.randint(0, len(self))
        return example


        
    
        
        
        
        
        