"""
Show registration results.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from tqdm import tqdm

from cvhelpers.torch_helpers import to_numpy
from my_models.regtr import RegTR
from utils.misc import load_config
from utils.se3_numpy import se3_transform

def load_point_cloud(fname, tran, transformation):
    if fname.endswith('.pth'):
        data = torch.load(fname)
    elif fname.endswith('.ply') or fname.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(fname)
        if tran:
            pcd = pcd.transform(transformation)
        data = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals).astype(np.float32)
    elif fname.endswith('.bin'):
        data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    else:
        raise AssertionError('Cannot recognize point cloud format')
    return data[:, :3], normals  # ignore reflectance, or other features if any

def generate_transform():
    rot_mag = 45
    trans_mag = 0.5
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
    rand_SE3 = np.vstack((rand_SE3,np.array([0,0,0,1])))
    # print("随机矩阵\n", np.linalg.inv(rand_SE3))
    return rand_SE3

def compute_relative_errors(T_true, T_est):
    # 提取真实旋转矩阵和估计的旋转矩阵
    R_true = T_true[:3, :3]
    R_est = T_est[:3, :3]

    # 计算相对旋转误差
    trace = np.trace(np.dot(R_true.T, R_est))
    trace = np.clip(trace, -1.0, 3.0)
    # print("trace: ",trace)
    rotation_error = abs(np.arccos((trace - 1.0) / 2.0) * 180/np.pi)

    # 提取真实平移向量和估计的平移向量
    t_true = T_true[:3, 3]
    t_est = T_est[:3, 3]

    # 计算相对平移误差
    translation_error = np.linalg.norm(t_true - t_est)

    return rotation_error, translation_error

def compute_chamfer_distance(pcd1, pcd2):
    # 计算点云1到点云2的距离
    dist1 = np.min(np.sqrt(np.sum((pcd1[:, None] - pcd2) ** 2, axis=-1)), axis=1).sum()

    # 计算点云2到点云1的距离
    dist2 = np.min(np.sqrt(np.sum((pcd2[:, None] - pcd1) ** 2, axis=-1)), axis=1).sum()

    # 计算倒角距离，即两个方向的距离之和
    chamfer_distance = dist1 + dist2
    chamfer_distance/=pcd1.shape[0]

    return chamfer_distance

def transform_point_cloud(point_cloud, transformation_matrix):
    # 将点云转换为齐次坐标形式（添加一列1）
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # 将位姿转换矩阵应用于点云
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T

    # 将变换后的点云转换为非齐次坐标形式（去除最后一列）
    transformed_points = transformed_points[:, :3]

    return transformed_points

#---------------------------------------------------------------------------------------------------------------------------------------

_examples = [
    
     # 0 use result of 01-23:glo+xyzsin
    ('../logs/CustomData/240123-glo+xyzsin/ckpt/model-169344.pth',
     '/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src',
     '/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar'),
     # 1 240124-geo+xyzsin
    ('../logs/CustomData/240124-geo+xyzSine/ckpt/model-117504.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar'),
     # 2 240126-same-circleLoss
    ('../logs/CustomData/240126-circleLoss/ckpt/model-138240.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--example', type=int, default=0,
                    help=f'Example pair to run (between 0 and {len(_examples) - 1})')
opt = parser.parse_args()



def main():
    # Retrieves the model and point cloud paths
    ckpt_path, src_path, tgt_path = _examples[opt.example]

    # Load configuration file
    cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Instantiate model and restore weights
    model = RegTR(cfg).to(device)
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

    # 获取文件夹中的所有文件
    src_files = sorted(Path(src_path).glob('*'))
    tgt_files = sorted(Path(tgt_path).glob('*'))

    assert len(src_files) == len(tgt_files), "The number of files in the source and target folders must be the same."
    RRE = []
    TRE = []
    CD = []

    num = len(src_files)
    np.random.seed(776)  # 设置种子
    # 循环处理每对点云
    for src_file, tgt_file in tqdm(zip(src_files, tgt_files), total=num):
        # Loads point cloud data: Each is represented as a Nx3 numpy array
        init_tran = generate_transform()
        src_xyz, src_normals = load_point_cloud(str(src_file), True, init_tran)
        tgt_xyz, tgt_normals = load_point_cloud(str(tgt_file), False, None)

        # Feeds the data into the model
        data_batch = {
            'src_xyz': [torch.from_numpy(src_xyz).float().to(device)],
            'tgt_xyz': [torch.from_numpy(tgt_xyz).float().to(device)],
            'src_normals': [torch.from_numpy(src_normals).float().to(device)],
            'tgt_normals': [torch.from_numpy(tgt_normals).float().to(device)],
        }
        outputs = model(data_batch)

        # Visualize the results
        b = 0
        pose = to_numpy(outputs['pose'][-1, b])
        src_kp = to_numpy(outputs['src_kp'][b])
        src2tgt = to_numpy(outputs['src_kp_warped'][b][-1])  # pred. corresponding locations of src_kp
        overlap_score = to_numpy(torch.sigmoid(outputs['src_overlap'][b][-1]))

        # 打印
        pose_ = np.vstack((pose,np.array([0,0,0,1])))
        # 计算相对旋转误差和相对平移误差
        rotation_error, translation_error = compute_relative_errors(np.linalg.inv(init_tran), pose_)
        RRE.append(rotation_error)
        # print("RRE:",rotation_error)
        TRE.append(translation_error)
        # 计算倒角距离
        chamfer_distance = compute_chamfer_distance(transform_point_cloud(src_xyz,pose_), tgt_xyz)
        CD.append(chamfer_distance)

    print("RRE:",np.mean(RRE))
    print("TRE:",np.mean(TRE))
    print("CD:",np.mean(CD))

    np.savetxt("../my_results/RRE.txt", RRE)
    np.savetxt("../my_results/TRE.txt", TRE)
    np.savetxt("../my_results/CD.txt", CD)


if __name__ == '__main__':
    main()