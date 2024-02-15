import torch
import torch.nn as nn

from utils.se3_torch import se3_transform_list, se3_transform
from utils.pcd_partition import point_to_node_partition, index_select

_EPS = 1e-6


class CorrCriterion(nn.Module):
    """Correspondence Loss.
    """
    def __init__(self, metric='mae'):
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric

    def forward(self, pts_before, kp_before, kp_warped_pred, pose_gt, overlap_weights=None):

        losses = {}
        B = pose_gt.shape[0]

        ## original implementation
        # kp_warped_gt = se3_transform_list(pose_gt, kp_before)
        # corr_err = torch.cat(kp_warped_pred, dim=0) - torch.cat(kp_warped_gt, dim=0)

        corr_err = []
        # get point to node partition: patch
        for b in range(B):
            _, node_masks, node_knn_indices = point_to_node_partition(
                points=pts_before[b], nodes=kp_before[b], point_limit=50) # node_knn_indices: (M, K)
            pts = se3_transform(pose_gt[b], pts_before[b]) # (N, 3)
            knn_points = index_select(pts, node_knn_indices, dim=0) # (M, K, 3)
            # knn_normals = index_select(points_normals[b], node_knn_indices, dim=0) # (M, K, 3)
            err = knn_points - kp_warped_pred[b].unsqueeze(1) # (M, K, 3)
            corr_err.append(err.mean(dim=-2)) # (M, 3)

        corr_err = torch.cat(corr_err, dim=0) # (M, 3)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)

        return mean_err

