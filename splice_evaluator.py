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
from eval.image_splice.metrics import F1_Metric, MCC_Metric, mAP_Metric

from datasets.columbia import ColumbiaDataset
from datasets.dso_1 import DSO_1_Dataset
from datasets.in_the_wild import InTheWildDataset
from datasets.realistic_tampering import RealisticTamperingDataset
from datasets.scene_completion import SceneCompletionDataset
from datasets.casia_2 import CASIA_2_Dataset
from datasets.casia_1 import CASIA_1_Dataset


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
        
        metric_classes = {
            "f1_score": F1_Metric(),
            "mcc": MCC_Metric(),
            "mAP": mAP_Metric(),
        }
        
        y_true = []
        y_score = []
        label_map = []
        score_map = []

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
            
            # Store ground-truths
            y_true.append(data["label"])
            label_map.append(data["map"])

            # Store predictions
            y_score.append(pred["score"])
            score_map.append(pred["ms"])

        # Compute patch average precision
        print("computing patch score ...")
        
        for type, mClass in metric_classes.items():
            self.metrics[type] = mClass.compute()
        
        y_true = np.array(y_true)
        label_map = np.stack(label_map, axis=0)

        y_score = np.array(y_score)
        score_map = np.stack(score_map, axis=0)
        
        # Compute localization metrics
        print("computing localization score ...")
        self._compute_cIoU(label_map, score_map)

        
        for k, v in self.metrics.items():
            print(k, "--", v)

        return self.metrics


    def _compute_cIoU(self, label_map, score_map):
        
        if np.isnan(score_map).any():
            print("WARNING: NaN values in localization prediction scores!")
            score_map[np.isnan(score_map)] = 0
            
        # Compute for spliced regions
        _, iou_spliced = self.find_optimal_threshold(score_map, label_map)
        iou_spliced = iou_spliced.mean().item()

        # Compute for non-spliced regions
        invert_label_map = 1 - label_map
        invert_score_map = 1 - score_map

        _, iou_non_spliced = self.find_optimal_threshold(
            invert_score_map, invert_label_map
        )
        iou_non_spliced = iou_non_spliced.mean().item()
 
        self.metrics["IoU-spliced"] = iou_spliced
        self.metrics["IoU-non-spliced"] = iou_non_spliced
        # Compute mean IoU
        self.metrics["IoU"] = (iou_spliced + iou_non_spliced) / 2
        print("cIoU", "--", self.metrics["IoU"] )
        

    @staticmethod
    def find_optimal_threshold(
        pred_mask: np.ndarray, groundtruth_masks: np.ndarray
    ):
        """https://codereview.stackexchange.com/questions/229341/pytorch-vectorized-implementation-for-thresholding-and-computing-jaccard-index

        Parameters
        ----------
        pred_mask : np.ndarray (float32)
            [B, H, W], range [0, 1], probability prediction map
        groundtruth_masks : np.ndarray (uint8)
            [B, H, W], values one of {0, 1}, binary label map

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            [B], optimal thresholds for each image
            [B], corresponding jaccard scores for each image
        """
        n_patch = groundtruth_masks.shape[0]

        groundtruth_masks_tensor = torch.from_numpy(groundtruth_masks)
        pred_mask_tensor = torch.from_numpy(pred_mask)

        # if USE_CUDA:
        #     groundtruth_masks_tensor = groundtruth_masks_tensor.cuda()
        #     pred_mask_tensor = pred_mask_tensor.cuda()

        vector_pred = pred_mask_tensor.view(n_patch, -1)
        vector_gt = groundtruth_masks_tensor.view(n_patch, -1)
        vector_pred, sort_pred_idx = torch.sort(vector_pred, descending=True)
        vector_gt = vector_gt[torch.arange(vector_gt.shape[0])[:, None], sort_pred_idx]
        gt_cumsum = torch.cumsum(vector_gt, dim=1)
        gt_total = gt_cumsum[:, -1].reshape(n_patch, 1)
        predicted = torch.arange(start=1, end=vector_pred.shape[1] + 1)
        # if USE_CUDA:
        #     predicted = predicted.cuda()
        gt_cumsum = gt_cumsum.type(torch.float)
        gt_total = gt_total.type(torch.float)
        predicted = predicted.type(torch.float)
        jaccard_idx = gt_cumsum / (gt_total + predicted - gt_cumsum)
        max_jaccard_idx, max_indices = torch.max(jaccard_idx, dim=1)
        max_indices = max_indices.reshape(-1, 1)
        best_threshold = vector_pred[
            torch.arange(vector_pred.shape[0])[:, None], max_indices
        ]
        best_threshold = best_threshold.reshape(-1)

        return best_threshold, max_jaccard_idx




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="your/pretrained/model/path")
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--data_name", default="in_the_wild")
    parser.add_argument("--data_base_path", default="/scratch/ahowens_root/ahowens1/neymar/test_dataset")
    parser.add_argument("--patch_size", type=int, default=124)
    parser.add_argument("--num_per_dim", type=int, default=25)
    args = parser.parse_args()
    
    if args.dataset not in ['in_the_wild', 'columbia', 'dso_1', 'realistic_tampering', 'scene_completion', 'casia_1', 'casia_2']: raise NotImplementedError
    
    print("[load wrapper model]")
    model, _ = load_wrapper_model(device=DEVICE, state_dict_path=args.ckpt_path, model_name="RN50", input_resolution=args.patch_size)

    
    if args.dataset == "in_the_wild": dataset = InTheWildDataset(root_dir=args.dataset_base_path)
    elif args.dataset == 'columbia': dataset = ColumbiaDataset(root_dir=args.dataset_base_path, spliced_only=False)
    elif args.dataset == "dso_1": dataset = DSO_1_Dataset(root_dir=args.dataset_base_path, spliced_only=False)
    elif args.dataset == 'realistic_tampering': dataset = RealisticTamperingDataset(root_dir=args.dataset_base_path)
    elif args.dataset == 'scene_completion': dataset = SceneCompletionDataset(root_dir=args.dataset_base_path)
    elif args.dataset == 'casia_2': dataset = CASIA_2_Dataset(root_dir=args.dataset_base_path)
    elif args.dataset == 'casia_1': dataset = CASIA_1_Dataset(root_dir=args.dataset_base_path)
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
                    