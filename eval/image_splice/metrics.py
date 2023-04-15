from typing import Callable
import pdb
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


class LocalizationMetric:
    def __init__(
        self, metric: Callable[[np.ndarray, np.ndarray], float], thresh=False
    ):
        self.metric = metric
        # Whether the metric takes in binary decision maps
        self.thresh = thresh

        self.scores = []

    def update(self, label_map: np.ndarray, score_map: np.ndarray):
        # FIXME Search for the optimal threshold?
        if self.thresh:
            pred_map = (score_map > 0.5).astype(int)
        else:
            pred_map = score_map
        score = self.metric(label_map.flatten(), pred_map.flatten())

        # Consider inverted map
        inverted_map = 1 - score_map
        if self.thresh:
            pred_map = (inverted_map > 0.5).astype(int)
        else:
            pred_map = inverted_map

        inverted_score = self.metric(label_map.flatten(), pred_map.flatten())

        # Take the better score
        self.scores.append(max(score, inverted_score))
        return max(score, inverted_score)

    def compute(self):
        return sum(self.scores) / len(self.scores)


class mAP_Metric(LocalizationMetric):
    def __init__(self):
        super().__init__(average_precision_score)


class F1_Metric(LocalizationMetric):
    def __init__(self):
        # Compute optimal f1 score
        def optimal_f1(y_true, y_score):
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            f1_scores = 2 * recall * precision / (recall + precision)
            # Account for nan values
            f1_scores[np.isnan(f1_scores)] = 0

            return f1_scores.max()

        super().__init__(optimal_f1)


class MCC_Metric(LocalizationMetric):
    def __init__(self):
        super().__init__(matthews_corrcoef, thresh=True)


class AUC_Metric(LocalizationMetric):
    def __init__(self):
        super().__init__(roc_auc_score)