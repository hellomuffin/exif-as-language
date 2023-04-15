
from pathlib import Path
import sys
import cv2
import numpy as np
import scipy
import sklearn.cluster
from sklearn.decomposition import PCA
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop
import torchvision.transforms as T
import random
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from utils.structures import PatchedImage 


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform():
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def preprocess(image: torch.Tensor):
    toPIL = T.ToPILImage()
    image = toPIL(image)
    func =  _transform()
    return func(image)


def create_logits(x1, x2):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 =  x1 @ x2.t()
    return logits_per_x1



def mean_shift(points_, heat_map, window, iter):
    points = np.copy(points_)
    kdt = scipy.spatial.cKDTree(points)   
    eps_5 = np.percentile(
        scipy.spatial.distance.cdist(points, points, metric="euclidean"), window
    )
    for epis in range(iter):
        for point_ind in range(points.shape[0]):
            point = points[point_ind]
            nearest_inds = kdt.query_ball_point(point, r=eps_5)
            points[point_ind] = np.mean(points[nearest_inds], axis=0)
    val = []
    for i in range(points.shape[0]):
        val.append(
            kdt.count_neighbors(scipy.spatial.cKDTree(np.array([points[i]])), r=eps_5)
        )
    mode_ind = np.argmax(val)
    ind = np.nonzero(val == np.max(val))
    return np.mean(points[ind[0]], axis=0).reshape(heat_map.shape[0], heat_map.shape[1])


def normalized_cut(res):
    res = 1 - res
    sc = sklearn.cluster.SpectralClustering(
        n_clusters=2, n_jobs=-1, affinity="precomputed"
    )
    out = sc.fit_predict(res.reshape((res.shape[0] * res.shape[1], -1)))
    vis = out.reshape((res.shape[0], res.shape[1]))
    return vis


class LinearHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear=torch.nn.Linear(input_dim, output_dim)

    def forward(self,x):
        x = x.to(torch.float32)
        x = self.linear(x)
        return x


class EXIF_SC:
    

    def __init__(
        self,
        model, 
        patch_size=128, 
        num_per_dim=30, 
        device="cuda:0", 
        linear_head=None, 
        ms_window = 10, 
        ms_iter = 5, 
        isOriginal=False
    ):
        """
        Parameters
        ----------
        model: CLIP_MODEL
            wrapped up clip model
        patch_size : int, optional
            Size of patches, by default 128
        num_per_dim : int, optional
            Number of patches to use along the largest dimension, by default None (stride using patch_size)
        device : str, optional
            , by default "cuda:0"
        """
        random.seed(44)
        self.patch_size = patch_size
        self.num_per_dim = num_per_dim
        self.device = torch.device(device)
        self.ms_window, self.ms_iter = ms_window, ms_iter
        self.isOriginal = isOriginal
        self.linear_head = linear_head
        
        print("[window]", ms_window, "[iter]", ms_iter)


        self.net = model
        self.net.eval()
        self.net.to(device)





    def predict(
        self,
        img: torch.Tensor,
        feat_batch_size=32,  # Does not affect compute time much?
        pred_batch_size=1024,  # Affects up to a certain extent
        blue_high=True,
    ):
        """
        Parameters
        ----------
        img : torch.Tensor
            [C, H, W], range: [0, 255]
        feat_batch_size : int, optional
            , by default 32
        pred_batch_size : int, optional
            , by default 1024
        blue_high : bool
            , by default True

        Returns
        -------
        Dict[str, Any]
            ms : np.ndarray (float32)
                Consistency map, [H, W], range [0, 1]
            ncuts : np.ndarray (float32)
                Localization map, [H, W], range [0, 1]
            score : float
                Prediction score, higher indicates existence of manipulation
        """
        _, height, width = img.shape
        assert (
            min(height, width) > self.patch_size
        ), "Image must be bigger than patch size!"
        
        # Initialize image and attributes
        img = self.init_img(img, num_per_dim=self.num_per_dim)
        self.img = img
        # Precompute features for each patch
        with torch.no_grad():
            patch_features = self.get_patch_feats(img, batch_size=feat_batch_size)

        # PCA visualization
        pca = PCA(n_components=3,whiten=True)
        feature_transform = pca.fit_transform(patch_features.cpu().numpy())
        pred_pca_map = self._predict_pca_map(
            img, feature_transform, batch_size=pred_batch_size
        ).numpy()
        
        
        # Predict consistency maps
        pred_maps = self._predict_consistency_maps(
            img, patch_features, batch_size=pred_batch_size
        ).detach().numpy()
        
        # sample prediction maps
        preds = [pred_maps[0,0], pred_maps[pred_maps.shape[0]-1,pred_maps.shape[1]-1], pred_maps[0,pred_maps.shape[1]-1], pred_maps[pred_maps.shape[0]-1,0], pred_maps[pred_maps.shape[0]//2,pred_maps.shape[1]//2], pred_maps[pred_maps.shape[0]-1,pred_maps.shape[1]//2], pred_maps[pred_maps.shape[0]*3//4,pred_maps.shape[1]//2], pred_maps[pred_maps.shape[0]*2//3,pred_maps.shape[1]//2]]
        
        
        # Produce a single response map
        ms = mean_shift(
            pred_maps.reshape((-1, pred_maps.shape[0] * pred_maps.shape[1])), pred_maps, window=self.ms_window, iter=self.ms_iter
        )

        # Run clustering to get localization map
        ncuts = normalized_cut(pred_maps)
        
        
        out_preds = []
        for pred in preds: out_preds.append(cv2.resize(pred, (width,height), interpolation=cv2.INTER_LINEAR))
        
        out_ms = cv2.resize(ms, (width,height), interpolation=cv2.INTER_LINEAR)
        out_ncuts = cv2.resize(
            ncuts.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        
        out_pca = np.zeros((height,width,3))
        p1, p3 = np.percentile(pred_pca_map,0.5), np.percentile(pred_pca_map, 99.5)
        pred_pca_map = (pred_pca_map - p1) / (p3-p1) * 255   # >0
        pred_pca_map[pred_pca_map<0] = 0
        pred_pca_map[pred_pca_map>255] = 255
        for i in range(3):
            out_pca[:,:,i] = cv2.resize(pred_pca_map[:,:,i], (width, height), interpolation=cv2.INTER_LINEAR)
        
        return {"pred_maps": pred_maps, "ms": out_ms, "ncuts": out_ncuts, "score": pred_maps.mean(), "preds":out_preds, "pca":out_pca, "affinity_matrix":self.generate_afinity_matrix(patch_features)}




    def init_img(self, img: torch.Tensor, num_per_dim):
        # Initialize image and attributes
        img = img.to(self.device)
        img = PatchedImage(img, self.patch_size, num_per_dim)

        return img


    def get_patch_consist_map(self, image, feat_batch_size, index, patch_fake):
        # Initialize image and attributes
        img = self.init_img(image, num_per_dim=None)
        # Precompute features for each patch
        with torch.no_grad():
            patch_features = self.get_patch_feats(img, batch_size=feat_batch_size) # [n_patches, n_features]
        
        # Predict consistency maps
        patch_consist_map = self.center_patch_consistency(patch_features, index, patch_fake)
        
        
        
        return patch_consist_map


    def center_patch_consistency(self, patch_features, index, patch_fake):
        center_feature = patch_features[index]
        
        cos_sims = create_logits(center_feature, patch_features)
        if patch_fake: 
            cos_sims = 1 - cos_sims
        return 1-cos_sims
        
        
        
    def _predict_consistency_maps(
        self, img: PatchedImage, patch_features: torch.Tensor, batch_size=64
    ):
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = torch.zeros(
            (
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
            )
        )
        # Number of predictions for each patch
        vote_counts = (
            torch.zeros(
                (
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                )
            )
            + 1e-4
        )

        # Perform prediction
        for idxs in img.pred_idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]
            patch_b_idxs = idxs[:, 2:]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = torch.from_numpy(
                np.ravel_multi_index(patch_a_idxs.T, [img.max_h_idx, img.max_w_idx])
            )  # [B]
            b_idxs = torch.from_numpy(
                np.ravel_multi_index(patch_b_idxs.T, [img.max_h_idx, img.max_w_idx])
            )

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 4096]
            b_feats = patch_features[b_idxs]

            sim = self.patch_similarity(a_feats, b_feats)

            # FIXME Is it possible to vectorize this?
            # Accumulate predictions for overlapping patches
            for i in range(len(sim)):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += sim[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += 1

        # Normalize predictions
        return responses / vote_counts
    
    
    def _predict_pca_map(
        self, img: PatchedImage, patch_features: torch.Tensor, batch_size=64
    ):
        if not img: img = self.img
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = torch.zeros(
            (
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
                3
            )
        )
        # Number of predictions for each patch
        vote_counts = (
            torch.zeros(
                (
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                    3
                )
            )
            + 1e-4
        )

        # Perform prediction
        for idxs in img.pca_idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = torch.from_numpy(
                np.ravel_multi_index(patch_a_idxs.T, [img.max_h_idx, img.max_w_idx])
            )  # [B]
  

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 3]


            # FIXME Is it possible to vectorize this?
            # Accumulate predictions for overlapping patches
            for i in range(a_feats.shape[0]):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :
                ] += a_feats[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :
                ] += 1

        # Normalize predictions
        return responses / vote_counts
    
    
    def patch_similarity(self, a_feats, b_feats):
            cos = create_logits(a_feats, b_feats).diagonal()
            cos = 1 - cos
            cos = cos.cpu()
            return cos
        




    def get_patch_feats(
        self, img: PatchedImage, batch_size=32
    ):
        """
        Get features for every patch in the image.
        Features used to compute if two patches share the same EXIF attributes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be fed into the network, by default 32

        Returns
        -------
        torch.Tensor
            [n_patches, 4096]
        """
        # Compute feature vector for each image patch
        patch_features = []

        # Generator for patches; raster scan order
        for patches in img.patches_gen(batch_size):
            processed_patches = torch.stack([preprocess(patch) for patch in patches],dim=0).to(self.device)
            feat = self.net.encode_image(processed_patches)

            if len(feat.shape) == 1:
                feat = feat.view(1, -1)
            patch_features.append(feat)

        # [n_patches, n_features]
        patch_features = torch.cat(patch_features, dim=0)

        return patch_features
    

    def generate_afinity_matrix(self, patch_features):
        patch_features = torch.nn.functional.normalize(patch_features)
        result = torch.matmul(patch_features, patch_features.t())

        # sort_idx = torch.argsort(result_norm)
        # result = result[sort_idx]
        # result = result[:, sort_idx]
        return result
    
    
    def get_valid_patch_mask(
        self, mask: PatchedImage, batch_size=32
    ):
        valid_mask = []
        for patches in mask.patches_gen(batch_size):
            patches = patches.reshape(patches.shape[0], -1) # [batch_size, patch_size * patch_size]
            patches_sum = torch.sum(patches, dim=1) # [batch_size]
            positive_mask = (patches_sum > self.patch_size*self.patch_size*0.9)
            negative_mask = (patches_sum == 0)
            valid_mask.append(positive_mask.long() - negative_mask.long())
        valid_mask = torch.cat(valid_mask, dim=0)
        return valid_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path",
        help="path to the weights file",
        default="artifacts/exif_sc.npy",
    )
    parser.add_argument(
        "--img_path",
        help="path to the input image file",
        default="data/demo.png",
    )
    args = parser.parse_args()

    model = EXIF_SC(args.weights_path)

    img = cv2.imread(args.img_path)[:, :, [2, 1, 0]]  # [H, W, C]
    img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
