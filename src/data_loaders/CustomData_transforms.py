"""Modified from RPMNet's transformation to compute correspondences and
groundtruth overlap
"""

import math
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import torch
import torch.utils.data
import open3d as o3d

from utils.se3_numpy import se3_transform, se3_inv
from utils.so3_numpy import so3_transform


# noinspection PyPep8Naming
class RandomTransformSE3_euler:
    """
    Generate a random SE3 transformation (3, 4) 
    Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def __init__(self, 
                 rot_mag: float = 180.0, 
                 trans_mag: float = 1.0, 
                 seed: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            seed (bool): If true, will seed the random number generator
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._seed = seed
    def generate_transform(self):

        if self._seed:
            np.random.seed(776)

        rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3
    
    def apply_transform(self, p0, n0, transform_mat):
        p1 = se3_transform(transform_mat, p0[:, :3])
        n1 = so3_transform(transform_mat[:3, :3], n0)
        igt = transform_mat
        gt = se3_inv(igt)

        return p1, n1, gt, igt

    def transform(self, tensor, normal):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, normal, transform_mat)


# my transforms----------------------------
class ReadPcd:
    """read pcd from .pcd"""
    def __call__(self, sample: Dict):
        sample['src_pcd'] = o3d.io.read_point_cloud(sample['src_pcd'])
        sample['src_normals'] = np.asarray(sample['src_pcd'].normals).astype(np.float32)
        sample['src_pcd'] = np.asarray(sample['src_pcd'].points).astype(np.float32)

        sample['tar_pcd'] = o3d.io.read_point_cloud(sample['tar_pcd'])
        sample['tar_normals'] = np.asarray(sample['tar_pcd'].normals).astype(np.float32)
        sample['tar_pcd'] = np.asarray(sample['tar_pcd'].points).astype(np.float32)        

        n_points = sample['src_pcd'].shape[0]
        sample['correspondences'] = np.tile(np.arange(n_points), (2, 1)) # (2, n_points)

        return sample
    
class Shuffle:
    """Shuffle the point cloud"""
    def __call__(self, sample: Dict):
        # shuffle
        src_index = np.arange(sample['src_pcd'].shape[0])
        tar_index = np.arange(sample['tar_pcd'].shape[0])
        np.random.shuffle(src_index)
        np.random.shuffle(tar_index)
        sample['src_pcd'] = sample['src_pcd'][src_index]
        sample['tar_pcd'] = sample['tar_pcd'][tar_index]
        sample['src_normals'] = sample['src_normals'][src_index]
        sample['tar_normals'] = sample['tar_normals'][tar_index]
        return sample

class RandomTransform(RandomTransformSE3_euler):
    def __call__(self, sample:Dict):
        src_transformed, src_normal_transformed, transform_r_s, _ = self.transform(sample['src_pcd'],
                                                           sample['src_normals'])
        sample['transform_gt'] = transform_r_s  # Apply to source to get reference
        sample['src_raw'] = sample.pop('src_pcd')
        sample['src_pcd'] = src_transformed
        sample['src_normals'] = src_normal_transformed
        return sample
    
class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.003, clip=0.01):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):
        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz
        return pts

    def __call__(self, sample):
        sample['src_pcd'] = self.jitter(sample['src_pcd'])
        sample['tar_pcd'] = self.jitter(sample['tar_pcd'])
        return sample


class Coorespondence_getter():
    def __init__(self):
        self.search_radius = 5
        self.K = 1 # one nearest point
    def __call__(self, sample:Dict):
        transf = np.vstack((sample['transform_gt'], np.array([0,0,0,1]))).astype(np.float32)
        src_ply = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['src_pcd']))
        tgt_ply = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample['tar_pcd']))
        src_ply.transform(transf)
        pcd_tree = o3d.geometry.KDTreeFlann(tgt_ply)
        src_npy = np.array(src_ply.points)
        corrs = []
        for i in range(src_npy.shape[0]):
            point = src_npy[i]
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, self.search_radius)
            if self.K is not None:
                idx = idx[:self.K]
            for j in idx:
                corrs.append([i, j])
        sample['correspondences'] = np.array(corrs).T
        src_overlap = np.zeros(sample['src_pcd'].shape[0], dtype=bool)
        tar_overlap = np.zeros(sample['tar_pcd'].shape[0], dtype=bool)
        src_overlap[sample['correspondences'][0]] = 1
        tar_overlap[sample['correspondences'][1]] = 1
        sample['ref_overlap'] = tar_overlap
        sample['src_overlap'] = src_overlap
        return sample
    
# ----------------------------------------------