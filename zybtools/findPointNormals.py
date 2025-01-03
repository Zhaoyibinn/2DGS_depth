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

import os
import sys
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from torch_kdtree import build_kd_tree
import matplotlib.pyplot as plt

from extra_models.curv_opt import curv_loss

import rerun as rr




def cal_rendered_normal_loss(points, normals):
    

    
    return normal_cross


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






def sphere(radius, num_points=100):

    r = radius
    # 定义极角和方位角的范围和步长
    theta = np.linspace(0, np.pi/2, num_points)  # 从0到pi/2，100个点
    phi = np.linspace(0, 2*np.pi, num_points)  # 从0到2pi，100个点

    # 使用球坐标到笛卡尔坐标的转换公式
    x = r * np.outer(np.sin(theta), np.cos(phi))
    y = r * np.outer(np.sin(theta), np.sin(phi))
    z = r * np.outer(np.cos(theta), np.ones_like(phi))

    # 将x, y, z坐标合并成点云
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    unique_points, idx = np.unique(points, axis=0, return_index=True)
    return unique_points

def plane(num_points=100):
    x = np.random.uniform(-10, 10, num_points)
    y = np.random.uniform(-10, 10, num_points)

    # z坐标始终为0，因为我们在xy平面上
    z = np.random.uniform(low=-0.5, high=0.5,size=num_points)
    # 将x, y, z坐标组合成一个NumPy数组
    points = np.column_stack((x, y, z))
    # new_row = [[-5,5,0.5],[-4.5,4.5,0.5],[4.5,4.5,0.5],[-4.5,-4.5,0.5],[-4.3,4.3,0.5]]
    # points = np.append(points,new_row,axis=0)
    
    return points

if __name__ == "__main__":

    cloud = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/rgbd_dataset_freiburg1_desk/sparse/0/points3D.ply")
    print(cloud)
    np_cloud = np.asarray(cloud.points)[::10,:]
    np_cloud = np.asarray(cloud.points)
    start = time.time()
    print(np_cloud.shape)
    pt_cloud = torch.tensor(np_cloud,dtype=torch.float32, device='cuda', requires_grad=True)

    




    num_iterations = 2500





    # point_cloud = sphere(1)
    point_cloud = plane(10000)
    # pt_cloud = torch.tensor(point_cloud,dtype=torch.float32, device='cuda', requires_grad=True)

    pcd_origin = o3d.geometry.PointCloud()
    pcd_origin.points = o3d.utility.Vector3dVector(np.array(pt_cloud.cpu().detach()))

    # all_curv,normals = findPointNormals(pt_cloud, 10)
    # loss = all_curv.mean()

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.array(pt_cloud.cpu().detach()))
    # pcd.normals = o3d.utility.Vector3dVector(np.array(normals.cpu().detach()))
    # o3d.visualization.draw_geometries([pcd],point_show_normal=True)

    optimizer = optim.SGD([pt_cloud], lr=10.00)
    y = []

    rr.init("viewer",spawn=True)
    rr.spawn(connect=False)
    rr.connect()
    for i in tqdm(range(num_iterations), desc='Processing'):
        optimizer.zero_grad()

        # all_curv,normals,normal_cross,dis_close = findPointNormals(pt_cloud, 10)
        # loss_normal = torch.abs(normal_cross).mean() 
        # loss_curv = all_curv.mean()
        # loss_dis = 1/dis_close.mean()
        # loss = loss_normal 
        
        loss = curv_loss(pt_cloud, 10)
        
        
        loss.backward()
        # print(pt_cloud.grad)
        end = time.time()
        # print("all_time: " , end - start)
        optimizer.step()
        tqdm.write(f"Iteration {i+1}, Loss: {loss.item():.6f},std: {torch.std(pt_cloud[:,-1]).item():.6f}")
        y.append(torch.std(pt_cloud[:,-1]).item())

        if i %1 ==0:
            rr.set_time_sequence("step", i)
            rr.log(f"pt/trackable", rr.Points3D(pt_cloud.cpu().detach(), radii=0.01))

    x = range(num_iterations)
    plt.plot(x,y)
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pt_cloud.cpu().detach()))
    o3d.visualization.draw_geometries([pcd_origin])
    o3d.visualization.draw_geometries([pcd])



# start = time.time()
# curvature = calculate_surface_curvature(cloud)
# print(curvature)
# end = time.time()

