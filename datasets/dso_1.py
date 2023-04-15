"""DSO-1 Dataset

- https://recodbr.wordpress.com/code-n-data/#dso1_dsi1
- T. J. d. Carvalho, C. Riess, E. Angelopoulou, H. Pedrini and A. d. R. Rocha, “Exposing Digital Image Forgeries by Illumination Color Classification,” in IEEE Transactions on Information Forensics and Security, vol. 8, no. 7, pp. 1182-1194, July 2013. doi: doi: 10.1109/TIFS.2013.2265677
"""
import zipfile
from pathlib import Path
from typing import Any, Dict
import pdb
import cv2
import numpy as np
import toml
import torch
from torch.utils.data import Dataset

METADATA_FILENAME = Path("data/raw/dso_1/metadata.toml")
DL_DATA_DIRNAME = Path("data/downloaded/dso_1")
PROCESSED_DATA_DIRNAME = DL_DATA_DIRNAME / "tifs-database"


class DSO_1_Dataset(Dataset):
    def __init__(self, root_dir=PROCESSED_DATA_DIRNAME, spliced_only=False):

        self.to_label = {"normal": 0, "splicing": 1}
        self.root_dir = Path(root_dir)

        # Get list of all image paths
        img_dir = self.root_dir / "DSO-1"
        self.img_paths = list(img_dir.glob("*.png"))
        # Filter out authentic images
        if spliced_only:
            self.img_paths = [
                p for p in self.img_paths if p.stem.split("-")[0] == "splicing"
            ]
        print("length", len(self.img_paths))
        dataset_len = 100 if spliced_only else 200
        assert (
            len(self.img_paths) == dataset_len
        ), "Incorrect expected number of images in dataset!"


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
        img_path = self.img_paths[idx]

        # Get image
        img = cv2.imread(str(img_path))[:, :, [2, 1, 0]]  # [H, W, C]
        assert img.dtype == np.uint8, "Image should be of type int!"
        assert (
            img.min() >= 0 and img.max() <= 255
        ), "Image should be bounded between [0, 255]!"

        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        # Get label
        cat = img_path.stem.split("-")[0]
        label = self.to_label[cat]

        # Get spliced map
        map_dir = self.root_dir / "DSO-1-Fake-Images-Masks"
        _, height, width = img.shape

        if label:
            img_name = img_path.name
            map_path = map_dir / img_name
            map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)
            assert map.dtype == np.uint8, "Ground-truth should be of type int!"
            assert (
                map.min() >= 0 and map.max() <= 255
            ), "Ground-truth should be bounded between [0, 255]!"

            # Resize map if doesn't match image
            if (height, width) != map.shape:
                map = cv2.resize(map, (width, height), interpolation=cv2.INTER_LINEAR)

            map[map > 0] = 1
            
            map[map==1] = 2
            map[map==0] = 1
            map[map==2] = 0

        # If authentic image
        else:
            map = np.zeros((height, width), dtype=np.uint8)
        
        map = torch.tensor(map)
        
        img_name = img_path.name
        
        return {"img": img, "label": label, "map": map, "name": img_name, "path": img_path}

    def __len__(self):
        return len(self.img_paths)
