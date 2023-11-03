from typing import Any, Dict, Tuple
import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from tqdm import tqdm

from model_wrapper import load_wrapper_model
from eval.image_splice.exif_sc import EXIF_SC
from eval.image_splice.metrics import  mAP_Metric, cIoU_Metric

from datasets.columbia import ColumbiaDataset
from datasets.dso_1 import DSO_1_Dataset
from datasets.in_the_wild import InTheWildDataset
from datasets.realistic_tampering import RealisticTamperingDataset
from datasets.scene_completion import SceneCompletionDataset
from datasets.casia_2 import CASIA_2_Dataset
from datasets.casia_1 import CASIA_1_Dataset
from datasets.general import GeneralDataset


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class Evaluator:
    def __init__(self, 
            model, 
            dataset: Dataset, 
            patch_size: int, 
            device: str = 'cuda:0', 
            result_dir = "",  
            linear_head=None,
            ms_window=10, 
            ms_iter=5, 
            isOriginal=False
        ):
        self.isOriginal = isOriginal
        self.exif_sc = EXIF_SC(
            model=model, 
            patch_size=patch_size, 
            device=device, 
            linear_head=linear_head,
            ms_window=ms_window, 
            ms_iter=ms_iter, 
            isOriginal=isOriginal
        )
        self.dataset = dataset
        self.result_dir = result_dir

        self.metrics = {}

    def evaluate(self, resize: Tuple[int, int] = None):
        """
        Parameters
        ----------
        save : bool, optional
            Whether to save prediction arrays, by default False
        resize : Tuple[int, int], optional
            [H, W], whether to resize images / maps to a consistent shape

        Returns
        -------
        Dict[str, Any]
            AP : float
                Average precision score, for detection
            IoU : float
                Class-balanced IoU, for localization
            f1_score : float
                for localization
            mcc : float
                Matthews Correlation Coefficient, for localization
            mAP : float
                Mean Average Precision, for localization
            auc : float
                Area under the Receiving Operating Characteristic Curve, for localization
        """
        
        metric_classes = {"mAP": mAP_Metric()}

        for i in tqdm(range(len(self.dataset))):
            data = self.dataset[i]
            
            fname = data["name"]
            
            print("[count]", i, "[evaluating]", data["name"], "[shape]", data['img'].shape)
            
            pred = self.exif_sc.predict(data["img"])
            
            # Account for NaN values
            if np.isnan(pred["ms"]).any():
                print("WARNING: NaN values in localization prediction scores!")
                pred["ms"][np.isnan(pred["ms"])] = 0

            if np.isnan(pred["score"]):
                print("WARNING: NaN values in detection prediction scores!")
                pred["score"] = 0
                
            # Perform per-image evaluations
            for name, mClass in metric_classes.items():
                mClass.update(data["map"], pred["ms"])
                
            if resize:
                data["map"] = cv2.resize(
                    data["map"].numpy(), resize, interpolation=cv2.INTER_LINEAR
                )
                pred["ms"] = cv2.resize(
                    pred["ms"], resize, interpolation=cv2.INTER_LINEAR
                )
                
            # save predition
            sns.heatmap(pred["ms"], cmap="coolwarm", vmin=pred['ms'].min(), vmax=pred['ms'].max(), xticklabels=False, yticklabels=False, cbar=False)
            plt.savefig(os.path.join(self.result_dir,fname))
            plt.clf()

        # Compute patch average precision
        print("computing patch score ...")
        
        for type, mClass in metric_classes.items():
            self.metrics[type] = mClass.compute()
        
        for k, v in self.metrics.items():
            print(k, "--", v)

        return self.metrics



        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="your/pretrained/model/path")
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--dataset", default="in_the_wild")
    parser.add_argument("--dataset_base_path", default="/scratch/ahowens_root/ahowens1/neymar/test_dataset")
    parser.add_argument("--patch_size", type=int, default=124)
    parser.add_argument("--splice_only", default=False, action='store_true')
    parser.add_argument("--num_per_dim", type=int, default=25)
    args = parser.parse_args()
    
    if args.dataset not in ['in_the_wild', 'columbia', 'dso_1', 'realistic_tampering', 'scene_completion', 'casia_1', 'casia_2']: raise NotImplementedError
    
    print("[load wrapper model]")
    model, _ = load_wrapper_model(device=DEVICE, state_dict_path=args.ckpt_path, model_name="RN50", input_resolution=args.patch_size)

    dataset_base_path = os.path.join(args.dataset_base_path, args.dataset)
    if args.dataset == "in_the_wild": dataset = InTheWildDataset(root_dir=dataset_base_path)
    elif args.dataset == 'columbia': dataset = ColumbiaDataset(root_dir=dataset_base_path, spliced_only=args.splice_only)
    elif args.dataset == "dso_1": dataset = DSO_1_Dataset(root_dir=args.dataset_base_path, spliced_only=args.splice_only)
    elif args.dataset == 'realistic_tampering': dataset = RealisticTamperingDataset(root_dir=dataset_base_path)
    elif args.dataset == 'scene_completion': dataset = SceneCompletionDataset(root_dir=dataset_base_path)
    elif args.dataset == 'casia_2': dataset = CASIA_2_Dataset(root_dir=dataset_base_path)
    elif args.dataset == 'casia_1': dataset = CASIA_1_Dataset(root_dir=dataset_base_path)
    else: raise NotImplementedError
    

    print(f"----------- {args.dataset}------------- ")
    evaluator = Evaluator(
        model=model, 
        dataset=dataset, 
        patch_size=args.patch_size, 
        device=DEVICE, 
        result_dir=args.result_dir, 
        ms_window=10, 
        ms_iter=5,
    )
    evaluator.evaluate(resize=(1600,900))
                    