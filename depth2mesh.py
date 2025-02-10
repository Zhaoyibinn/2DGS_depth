import open3d as o3d
import numpy as np
import cv2
import os

def read_images_txt(file_path):
    images = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 12:
            continue
        img_id = int(parts[0])
        qvec = [float(x) for x in parts[1:5]]
        tvec = [float(x) for x in parts[5:8]]
        cam_id = int(parts[8])
        img_name = parts[-1]
        images[img_id] = {"qvec": qvec, "tvec": tvec, "cam_id": cam_id, "img_name": img_name}
    return images

# 初始化 TSDF 体积
def init_tsdf_volume(voxel_length, sdf_trunc, color_type):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,  # 体素大小，单位为米
        sdf_trunc=sdf_trunc,  # 截断 SDF 值
        color_type=color_type  # 颜色类型
    )

# 读取深度图和相机内参
def read_depth_image(depth_path):
    depth_im = cv2.imread(depth_path, -1).astype(float)
    depth_im /= 1000.0  # 将深度值从毫米转换为米
    return depth_im

# 创建 RGBD 图像
def create_rgbd_image(color_im, depth_im):
    color_im = o3d.geometry.Image(np.asarray(color_im, order="C", dtype=np.uint8))
    depth_im = o3d.geometry.Image(depth_im)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_im, depth_im, depth_scale=1.0, depth_trunc=0.5, convert_rgb_to_intensity=False
    )

# 主函数
def main():
    # 设置参数
    voxel_length = 0.01  # 体素大小为 1cm
    sdf_trunc = 0.02  # 截断 SDF 值为 2cm
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8

    # 初始化 TSDF 体积
    tsdf_volume = init_tsdf_volume(voxel_length, sdf_trunc, color_type)

    # 相机内参（需要根据实际情况修改）
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480, fx=517.3, fy=516.5, cx=318.6, cy=255.3
    )

    read_images_txt()

    # 深度图路径（假设深度图和彩色图在同一目录下）
    depth_dir = "data/tum/rgbd_dataset_freiburg1_floor_colmap/depth"
    color_dir = "data/tum/rgbd_dataset_freiburg1_floor_colmap/images"
    depth_files = sorted(os.listdir(depth_dir))
    color_files = sorted(os.listdir(color_dir))


    # 遍历所有深度图和彩色图
    for depth_file, color_file in zip(depth_files, color_files):
        depth_path = os.path.join(depth_dir, depth_file)
        color_path = os.path.join(color_dir, color_file)

        # 读取深度图和彩色图
        depth_im = read_depth_image(depth_path)
        color_im = cv2.imread(color_path)

        # 创建 RGBD 图像
        rgbd_im = create_rgbd_image(color_im, depth_im)

        # 假设相机位姿为单位矩阵（实际中需要提供相机位姿）
        cam_pose = np.eye(4)

        # 将 RGBD 图像整合到 TSDF 体积中
        tsdf_volume.integrate(rgbd_im, intrinsics, np.linalg.inv(cam_pose))

    # 从 TSDF 体积中提取点云或网格
    pcd = tsdf_volume.extract_point_cloud()
    mesh = tsdf_volume.extract_triangle_mesh()

    # 保存点云和网格
    o3d.io.write_point_cloud("output.ply", pcd)
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

if __name__ == "__main__":
    main()