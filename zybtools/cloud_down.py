import open3d as o3d
import numpy as np

in_path = "/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/tum/rgbd_dataset_freiburg1_desk_back/sparse/0/points3D_depth.ply"
out_path = "/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/tum/rgbd_dataset_freiburg1_desk_back/sparse/0/points3D_depth_down.ply"
# 读取PLY文件
pcd = o3d.io.read_point_cloud(in_path)

# 体素降采样
# 假设体素大小为0.05，你可以根据需要调整这个参数
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.015)

print(pcd)
print(voxel_down_pcd)

# 保存降采样后的点云到新的PLY文件
o3d.io.write_point_cloud(out_path, voxel_down_pcd)