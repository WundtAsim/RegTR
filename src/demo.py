"""Barebones code demonstrating REGTR's registration. We provide 2 demo
instances for each dataset

Simply download the pretrained weights from the project webpage, then run:
    python demo.py EXAMPLE_IDX
where EXAMPLE_IDX can be a number between 0-5 (defined at line 25)

The registration results will be shown in a 3D visualizer.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
import matplotlib

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from cvhelpers.torch_helpers import to_numpy
from my_models.regtr import RegTR
from utils.misc import load_config
from utils.se3_numpy import se3_transform


parser = argparse.ArgumentParser()
parser.add_argument('--example', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Controls viusalization of keypoints outside overlap region.')
parser.add_argument('--num', type=int, default=0)
opt = parser.parse_args()

_examples = [
    # 0 231230-
    ('../logs/CustomData/231230/ckpt/model-580608.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
     # 1 240119-811-right-overlap
    ('../logs/CustomData/240119-811-right-overlap/ckpt/model-345600.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
     # 2 240123-glo+xyzsin
    ('../logs/CustomData/240123-glo+xyzsin/ckpt/model-169344.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
     # 3 240124-geo+xyzsin
    ('../logs/CustomData/240124-geo+xyzSine/ckpt/model-117504.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
     # 4 240126-geo+xyzsin-circleLoss
    ('../logs/CustomData/240126-circleLoss/ckpt/model-138240.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
     # 5 240127-PPFglobal+xyzsin
    ('../logs/CustomData/240127-PPF+xyzSine/ckpt/model-152064.pth',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/src/src_96_left_{opt.num}.pcd',
     f'/media/yangqi/Windows-SSD/Users/Lenovo/Git/dataset/CustomData/train_val/test_data/tar/tar_96_left_{opt.num}.pcd'),
]


def visualize_result(src_xyz: np.ndarray, tgt_xyz: np.ndarray,
                     src_kp: np.ndarray, src2tgt: np.ndarray,
                     src_overlap: np.ndarray,
                     pose: np.ndarray,
                     threshold: float = 0.5):
    """Visualizes the registration result:
       - Top-left: Source point cloud and keypoints
       - Top-right: Target point cloud and predicted corresponding kp positions
       - Bottom-left: Source and target point clouds before registration
       - Bottom-right: Source and target point clouds after registration

    Press 'q' to exit.

    Args:
        src_xyz: Source point cloud (M x 3)
        tgt_xyz: Target point cloud (N x 3)
        src_kp: Source keypoints (M' x 3)
        src2tgt: Corresponding positions of src_kp in target (M' x 3)
        src_overlap: Predicted probability the point lies in the overlapping region
        pose: Estimated rigid transform
        threshold: For clarity, we only show predicted overlapping points (prob > 0.5).
                   Set to 0 to show all keypoints, and a larger number to show
                   only points strictly within the overlap region.
    """

    large_pt_size = 4
    color_mapper = matplotlib.pyplot.cm.ScalarMappable(norm=None, cmap=matplotlib.colormaps.get_cmap('coolwarm'))
    overlap_colors = (color_mapper.to_rgba(src_overlap[:, 0])[:, :3] * 255).astype(np.uint8)
    m = src_overlap[:, 0] > threshold
    # test overlap------------------------------------
    # print("threshord:",threshold)
    # print("src_overlap:",src_overlap[:,0])
    # print("m:",m)
    # test------------------------------------

    vis = cvv.Visualizer(
        win_size=(1600, 1000),
        num_renderers=4)

    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=0
    )
    vis.add_object(
        cvv.create_point_cloud(src_kp[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=0
    )

    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1
    )
    vis.add_object(
        cvv.create_point_cloud(src2tgt[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=1
    )

    # Before registration
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=2
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2
    )

    # After registration
    vis.add_object(
        cvv.create_point_cloud(se3_transform(pose, src_xyz), colors=colors.RED),
        renderer_idx=3
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3
    )

    vis.set_titles(['Source point cloud (with keypoints)',
                    'Target point cloud (with predicted source keypoint positions)',
                    'Before Registration',
                    'After Registration'])

    vis.reset_camera()
    vis.start()


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
    print("随机矩阵\n", np.linalg.inv(rand_SE3))
    return rand_SE3

def compute_relative_errors(T_true, T_est):
    # 提取真实旋转矩阵和估计的旋转矩阵
    R_true = T_true[:3, :3]
    R_est = T_est[:3, :3]

    # 计算相对旋转误差
    trace = np.trace(np.dot(R_true.T, R_est))
    rotation_error = np.arccos((trace - 1.0) / 2.0) * 180/np.pi

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

    # set seed
    np.random.seed(776)
    # Loads point cloud data: Each is represented as a Nx3 numpy array
    init_tran = generate_transform()
    src_xyz, src_normals = load_point_cloud(src_path, True, init_tran)
    tgt_xyz, tgt_normals = load_point_cloud(tgt_path, False, None)

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
    print("pose",pose_)
    # 计算相对旋转误差和相对平移误差
    rotation_error, translation_error = compute_relative_errors(np.linalg.inv(init_tran), pose_)
    print("相对旋转误差（RRE）:", rotation_error, "度")
    print("相对平移误差（TRE）:", translation_error, "m")
    # 计算倒角距离
    chamfer_distance = compute_chamfer_distance(transform_point_cloud(src_xyz,pose_), tgt_xyz)
    print("倒角距离:", chamfer_distance)

    visualize_result(src_xyz, tgt_xyz, src_kp, src2tgt, overlap_score, pose,
                     threshold=opt.threshold)


if __name__ == '__main__':
    main()