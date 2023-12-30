"""Data loader for ModelNet40
"""
import argparse, os, torch, h5py, torchvision
from typing import List

import numpy as np
from torch.utils.data import Dataset

from . import CustomData_transforms as Transforms


def get_train_datasets(args: argparse.Namespace):
    train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    train_data = CustomData(args, args.root, subset='train',
                             transform=train_transforms)
    val_data = CustomData(args, args.root, subset='val',
                           transform=val_transforms)

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    test_data = CustomData(args, args.root, subset='test',
                            transform=test_transforms)

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'custom'.

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """
    if noise_type == "clean":
        # 1-1 correspondence for each shuffled point, no noise
        train_transforms = [Transforms.ReadPcd(),
                            Transforms.RandomTransform(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Coorespondence_getter()]

        test_transforms = [Transforms.ReadPcd(),
                            Transforms.RandomTransform(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Coorespondence_getter()]
        
    elif noise_type == "custom":
        # shuffle + random transform + jitter
        train_transforms = [Transforms.ReadPcd(),
                            Transforms.RandomTransform(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.RandomJitter(),
                            Transforms.Coorespondence_getter()]

        test_transforms = [Transforms.ReadPcd(),
                            Transforms.RandomTransform(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.RandomJitter(),
                            Transforms.Coorespondence_getter()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class CustomData(Dataset):
    def __init__(self, args, root: str, subset: str = 'train', categories: List = None, transform=None):
        """Customed Dataset of leg point cloud.

        Args:
            root (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = args
        self._root = root
        self.n_in_feats = args.in_feats_dim
        self.overlap_radius = args.overlap_radius

        assert os.path.exists(os.path.join(root))

        dirname = subset + "_data"
        path = os.path.join(self._root, dirname, 'src')
        self._src_files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        path = os.path.join(root, dirname, 'tar')
        self._tmpl_files = [os.path.join(path, item) for item in sorted(os.listdir(path))]

        self._transform = transform

    def __getitem__(self, item):
        sample = {
            'src_pcd':self._src_files[item],
            'tar_pcd':self._tmpl_files[item],
            'idx': np.array(item, dtype=np.int32)
        }

        # -----------------------------------------
        # Apply perturbation
        if self._transform:
            sample = self._transform(sample)

        corr_xyz = np.concatenate([
            sample['src_pcd'][sample['correspondences'][0], :3],
            sample['tar_pcd'][sample['correspondences'][1], :3]], axis=1)

        # Transform to my format
        sample_out = {
            'src_xyz': torch.from_numpy(sample['src_pcd'][:, :3]),
            'tgt_xyz': torch.from_numpy(sample['tar_pcd'][:, :3]),
            'tgt_raw': torch.from_numpy(sample['src_raw'][:, :3]),
            'src_overlap': torch.from_numpy(sample['src_overlap']),
            'tgt_overlap': torch.from_numpy(sample['ref_overlap']),
            'correspondences': torch.from_numpy(sample['correspondences']),
            'pose': torch.from_numpy(sample['transform_gt']),
            'idx': torch.from_numpy(sample['idx']),
            'corr_xyz': torch.from_numpy(corr_xyz), 
        }

        return sample_out

    def __len__(self):
        return len(self._src_files)
