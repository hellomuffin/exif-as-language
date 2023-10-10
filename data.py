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
    def __init__(self, root_path, tag_threshold=0.49, resolution=124, ):
        super().__init__()
        self.exif_paths = glob.glob(f"{root_path}/*.json")
        
        self.resolution = resolution
        print("dataset length: ", len(self))
        
        with open('dataProcess/all_exif_count.json', 'r') as file:
            all_exif_count = json.load(file)
            self.filtered_keys = [key for key, value in all_exif_count.items() if value >= tag_threshold]
    
    def __len__(self):
        return len(self.image_paths)
    
    
    def get_possible_item(self, index):
        imagepath = self.exif_paths[index].replace('json', 'png')  # or jpg. depending on downlaoded format
        image = Image.open(imagepath)
        image = preprocess(image, self.resolution)

        with open(self.exif_paths[index]) as f:
            exif_dict = json.load(f)['exif']
            filtered_exif_dict = {}
            for k, v in exif_dict.items():
                if k in self.filtered_keys: filtered_exif_dict[k] = v
            exif_str = ", ".join([f"{key}: {value}" for key, value in filtered_exif_dict.items()])
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


        
    
        
        
        
        
        