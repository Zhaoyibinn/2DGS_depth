import open3d as o3d
import numpy as np

# 加载网格文件（例如 .ply 格式）
mesh_file_path = "/home/zhaoyibin/3DRE/instant-ngp/output/desk_3000_depth.ply"  # 替换为你的网格文件路径
mesh = o3d.io.read_triangle_mesh(mesh_file_path)

# 计算网格的顶点法线（可选）
mesh.compute_vertex_normals()

# 获取网格的顶点信息
vertices = np.asarray(mesh.vertices)
colors = np.asarray(mesh.vertex_colors)

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="Mesh to PCD")

# 保存点云为 .pcd 文件
output_pcd_path = "/home/zhaoyibin/3DRE/instant-ngp/output/desk_3000_depth_pcd.ply"  # 输出文件路径
o3d.io.write_point_cloud(output_pcd_path, pcd)
print(f"Point cloud saved to {output_pcd_path}")