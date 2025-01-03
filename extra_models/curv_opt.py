import numpy as np
from scipy.spatial import KDTree
# from numba import jit
import copy
import open3d as o3d
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm

from torch_kdtree import build_kd_tree
import matplotlib.pyplot as plt

import rerun as rr

def findPointNormals(pcd, numNeighbours):
    numNeighbours = numNeighbours + 1
    N = pcd.shape[0]
    start_kd = time.time()
    # 假设 build_kd_tree 是一个返回 PyTorch KDTree 接口的函数
    torch_kdtree = build_kd_tree(pcd)
    dists, indices = torch_kdtree.query(pcd, nr_nns_searches=numNeighbours)

    # print("kd_time : ", time.time() - start_kd)
    
    indices = indices[:, 1:]

    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(np.array(pcd.cpu().detach()))
    # pcd_o3d.paint_uniform_color([0, 1, 0])
    # pcd_o3d.colors[100] = [0,0,0]
    # for i in np.array(indices.detach().cpu())[100]:
        
    #     pcd_o3d.colors[i] = [0,0,0]

    # o3d.visualization.draw_geometries([pcd_o3d])



    pts = pcd.repeat((numNeighbours-1), 1)
    p = pts - pts[indices.T.flatten(), :]
    p = p.reshape(numNeighbours-1, pcd.shape[0], 3)
    p = p.permute(1, 0, 2)  # (N, numNeighbours-1, 3)
    
    # 计算协方差矩阵的值
    C = torch.zeros((N, 6), device=pcd.device)
    C[:, 0] = torch.sum(p[:, :, 0] * p[:, :, 0], dim=1)
    C[:, 1] = torch.sum(p[:, :, 0] * p[:, :, 1], dim=1)
    C[:, 2] = torch.sum(p[:, :, 0] * p[:, :, 2], dim=1)
    C[:, 3] = torch.sum(p[:, :, 1] * p[:, :, 1], dim=1)
    C[:, 4] = torch.sum(p[:, :, 1] * p[:, :, 2], dim=1)
    C[:, 5] = torch.sum(p[:, :, 2] * p[:, :, 2], dim=1)
    C = C / (numNeighbours-1)

    # 构建批量协方差矩阵
    Cmat = torch.stack([
    torch.stack([C[:, 0], C[:, 1], C[:, 2]], dim=1),
    torch.stack([C[:, 1], C[:, 3], C[:, 4]], dim=1),
    torch.stack([C[:, 2], C[:, 4], C[:, 5]], dim=1)
], dim=1)

    # 计算特征值和特征向量
    eigen_value, eigen_vector = torch.linalg.eig(Cmat)
    eigen_value = eigen_value.real
    eigen_vector = eigen_vector.real

    # 按特征值大小，由小到大排序
    sort_idx = torch.argsort(eigen_value, dim=1)
    eigen_value = torch.gather(eigen_value, 1, sort_idx)
    eigen_vector = torch.gather(eigen_vector, 2, sort_idx.unsqueeze(-2).repeat(1, 3, 1))

    # 利用特征值计算曲率
    curv = eigen_value[..., -1] / torch.sum(eigen_value, dim=-1)
    normals= eigen_vector[:,:,0]

    all_curv = curv.unsqueeze(-1)
    # print("cal_time : ", time.time() - start_kd)

    normals_repeat = normals.repeat((numNeighbours-1), 1)

    normals_near_repeat = normals_repeat[indices.T.flatten(), :]

    normal_cross = torch.cross(normals_repeat,normals_near_repeat)

    pts_1 = pcd
    p_1 = pts_1 - pts_1[indices[:,0]]

    dis_close = torch.norm(p_1, p=2, dim=1)
    return 1/all_curv - 1,normals,normal_cross,dis_close


def curv_loss(points,kd_num):

    curv_pca , normals , normal_cross , dis_close = findPointNormals(points,kd_num)
    loss_normal = torch.abs(normal_cross).mean() 
    loss_pca = curv_pca.mean()
    loss_dis = 1/dis_close.mean()
    loss = loss_normal
    return loss