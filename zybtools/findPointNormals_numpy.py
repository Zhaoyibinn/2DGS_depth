import numpy as np
from scipy.spatial import KDTree
# from numba import jit

def findPointNormals(pcd, numNeighbours):
    # create kdtree
    kdtree = KDTree(pcd)
    # get nearest neighbours
    _, indices = kdtree.query(pcd, k=numNeighbours+1)
    # remove self
    indices = indices[:, 1:]
    # find difference in position from neighbouring points
    pts = np.tile(pcd, (numNeighbours, 1))
    p = pts - pts[indices.T.flatten(), :]
    p = p[:, np.newaxis, :].transpose(1, 0, 2)
    p = p.reshape(numNeighbours, pcd.shape[0], 3)
    p = p.transpose(1, 0, 2)
    # calculate values for covariance matrix
    C = np.zeros(shape=(pcd.shape[0], 6))
    C[:, 0] = np.sum(p[:, :, 0] * p[:, :, 0], axis=1)
    C[:, 1] = np.sum(p[:, :, 0] * p[:, :, 1], axis=1)
    C[:, 2] = np.sum(p[:, :, 0] * p[:, :, 2], axis=1)
    C[:, 3] = np.sum(p[:, :, 1] * p[:, :, 1], axis=1)
    C[:, 4] = np.sum(p[:, :, 1] * p[:, :, 2], axis=1)
    C[:, 5] = np.sum(p[:, :, 2] * p[:, :, 2], axis=1)
    C = C / numNeighbours

    # normals and curvature calculation
    normals = np.zeros_like(pcd)
    curvature = np.zeros(shape=(pcd.shape[0], 1))
    for i in range(pcd.shape[0]):
        Cmat = np.asarray([[C[i, 0], C[i, 1], C[i, 2]],
                           [C[i, 1], C[i, 3], C[i, 4]],
                           [C[i, 2], C[i, 4], C[i, 5]]
                          ])
        eigen_value, eigen_vector = np.linalg.eig(Cmat)
        # 按特征值大小，由小到达排序
        sort_idx = np.argsort(eigen_value)
        eigen_value = eigen_value[sort_idx]
        eigen_vector = eigen_vector[:, sort_idx]
        # 利用特征值计算曲率
        curv = np.min(eigen_value) / np.sum(eigen_value)
        # 保存法向量和曲率
        normals[i, :] = (eigen_vector[:, 0].reshape(1, 3))
        curvature[i, :] = curv

    return normals, curvature

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

import open3d as o3d
import time
cloud = o3d.io.read_point_cloud("/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/rgbd_dataset_freiburg1_desk/sparse/0/points3D.ply")
print(cloud)
np_cloud = np.asarray(cloud.points)[::10,:]

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)
# o3d.visualization.draw_geometries([pcd])

start = time.time()
normals, curvature = findPointNormals(np_cloud, 10)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np_cloud)
pcd.normals = o3d.utility.Vector3dVector(normals)
o3d.visualization.draw_geometries([pcd],point_show_normal=True)