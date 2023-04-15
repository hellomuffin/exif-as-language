"""Columbia Uncompressed Image Splicing Detection Evaluation Dataset

- https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/
- Detecting Image Splicing Using Geometry Invariants And Camera Characteristics Consistency, Yu-Feng Hsu, Shih-Fu Chang
"""
import tarfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import toml
import torch
from torch.utils.data import Dataset

DL_DATA_DIRNAME = Path("data/downloaded/columbia")
PROCESSED_DATA_DIRNAMES = [DL_DATA_DIRNAME / "4cam_auth", DL_DATA_DIRNAME / "4cam_splc"]


class ColumbiaDataset(Dataset):
    def __init__(self, root_dir=DL_DATA_DIRNAME, spliced_only=False):

        self.to_label = {"4cam_auth": 0, "4cam_splc": 1}
        root_dir = Path(root_dir)

        # Get list of all image paths
        self.img_paths = []

        # Grab authentic images
        if not spliced_only:
            auth_dir = root_dir / "4cam_auth"
            auth_paths = list(auth_dir.glob("*.tif"))
            assert (
                len(auth_paths) == 183
            ), "Incorrect expected number of authentic images in dataset!"

            self.img_paths.extend(auth_paths)

        # Grab spliced images
        splc_dir = root_dir / "4cam_splc"
        splc_paths = list(splc_dir.glob("*.tif"))
        assert (
            len(splc_paths) == 180
        ), "Incorrect expected number of spliced images in dataset!"

        self.img_paths.extend(splc_paths)


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
        BRIGHT_GREEN = np.array([0, 255, 0])
        REGULAR_GREEN = np.array([0, 200, 0])

        _, height, width = img.shape

        if label:
            img_name = img_path.stem
            map_fullname = img_name + "_edgemask.jpg"
            map_path = img_path.parent / "edgemask" / map_fullname
            map = cv2.imread(str(map_path))[:, :, [2, 1, 0]]  # [H, W, C]

            # FIXME Should I include bright red too?
            # Find spliced region, i.e. green regions
            binary_map = np.zeros((height, width), dtype=np.uint8)
            bright_green_mask = (map == BRIGHT_GREEN).all(axis=-1)
            regular_green_mask = (map == REGULAR_GREEN).all(axis=-1)
            binary_map[bright_green_mask | regular_green_mask] = 1

        # If authentic image
        else:
            binary_map = np.zeros((height, width), dtype=np.uint8)

        binary_map = torch.tensor(binary_map)
        
        img_name = img_path.name

        return {"img": img, "label": 1, "map": binary_map, "name": img_name, "path": img_path}

    def __len__(self):
        
        return len(self.img_paths)
