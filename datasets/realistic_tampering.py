"""Realistic Tampering Dataset

- http://pkorus.pl/downloads/dataset-realistic-tampering
- P. Korus & J. Huang, Multi-scale Analysis Strategies in PRNU-based Tampering Localization, IEEE Trans. Information Forensics & Security, 2017
- P. Korus & J. Huang, Evaluation of Random Field Models in Multi-modal Unsupervised Tampering Localization, Proc. of IEEE Int. Workshop on Inf. Forensics and Security, 2016
"""
import zipfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import toml
import torch
from torch.utils.data import Dataset

METADATA_FILENAME = Path("data/raw/realistic_tampering/metadata.toml")
DL_DATA_DIRNAME = Path("data/downloaded/realistic_tampering")
PROCESSED_DATA_DIRNAME = DL_DATA_DIRNAME / "data-images"


class RealisticTamperingDataset(Dataset):
    def __init__(self, root_dir=PROCESSED_DATA_DIRNAME):

        self.to_label = {"pristine": 0, "tampered-realistic": 1}
        root_dir = Path(root_dir)

        # Get list of all image paths
        self.img_paths = []

        folders = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]
        sub_folders = ["pristine", "tampered-realistic"]

        for folder in folders:
            for sub_folder in sub_folders:
                img_dir = root_dir / folder / sub_folder
                # Grab all .TIF images
                self.img_paths.extend(img_dir.glob("*.TIF"))

        assert (
            len(self.img_paths) == 440
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
        label = self.to_label[img_path.parent.name]

        # Get localization map
        if label:
            img_name = img_path.stem
            img_fullname = img_name + ".PNG"
            map_path = img_path.parent.parent / "ground-truth" / img_fullname
            map = cv2.imread(str(map_path), cv2.IMREAD_GRAYSCALE)  # [H, W]
            assert map.dtype == np.uint8, "Ground-truth should be of type int!"
            assert (
                map.min() >= 0 and map.max() <= 255
            ), "Ground-truth should be bounded between [0, 255]!"

            # Turn all greys into black
            map[map > 0] = 1

        # If clean image
        else:
            _, height, width = img.shape
            map = np.zeros((height, width), dtype=np.uint8)
        
        map = torch.tensor(map)
        
        img_name = img_path.name

        return {"img": img, "label": label, "map": map, "name": img_name, "path": img_path}

    def __len__(self):
        return len(self.img_paths)
