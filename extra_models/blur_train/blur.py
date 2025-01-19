import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R_scipy
import torch

def Bayes_fit(camera_list,vis=False):
    Ts = []
    [before_viewpoint_cam,viewpoint_cam,after_viewpoint_cam] = camera_list
    for camera in camera_list:
        R = np.transpose(camera.R)
        t = camera.T
        # 上面的Rt与colmap对齐，C2W
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t
        T = np.linalg.inv(T)
        # 这里变成w2c
        Ts.append(T)
        # print("end")
    if vis:
        vizualizer = o3d.visualization.Visualizer()
        vizualizer.create_window(width=1200, height=900)

    intrinsic = np.array(camera.K)
    view_width_px = int(intrinsic[0][-1] * 2)
    view_height_px = int(intrinsic[1][-1] * 2)
    for pose in Ts:
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=view_width_px,
            view_height_px=view_height_px,
            intrinsic=intrinsic,
            extrinsic=np.linalg.inv(pose),
            scale=0.1
        )
        if vis:           
            vizualizer.add_geometry(camera_lines)
    if vis:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(Ts)[:,:3,3])
        vizualizer.add_geometry(pcd)
    
    translations = np.array([T[:3, 3] for T in Ts])
    rotations = np.array([T[:3, :3] for T in Ts])
    
    # 将旋转矩阵转换为旋转向量
    rotation_vectors = np.array([R_scipy.from_matrix(rot).as_rotvec() for rot in rotations])
    
    t = np.linspace(0, 1, len(Ts))
    cs_rot = CubicSpline(t, rotation_vectors, axis=0)
    cs_trans = CubicSpline(t, translations, axis=0)

    Bayes_pose = []
    
    for i in range(3):
        t = 0.5 + (i - 1)/10
        translation_interpolated = cs_trans(t)
        rotation_vector_interpolated = cs_rot(t)
        rotation_interpolated = R_scipy.from_rotvec(rotation_vector_interpolated).as_matrix()
        T_interpolated = np.eye(4)
        T_interpolated[:3, :3] = rotation_interpolated
        T_interpolated[:3, 3] = translation_interpolated

        Bayes_pose.append(T_interpolated)
        if vis:
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=view_width_px,
                view_height_px=view_height_px,
                intrinsic=intrinsic,
                extrinsic=np.linalg.inv(T_interpolated),
                scale=0.1
            )
                                
            vizualizer.add_geometry(camera_lines)
    if vis:
        vizualizer.run()
        vizualizer.destroy_window()

    return np.array(Bayes_pose) 
    # 返回的还是w2c


def mix_pic(pic_list,mix_weight):
    pic_tensor = torch.stack(pic_list, dim=0)
    weights_expanded = torch.abs(mix_weight.view(-1, 1, 1, 1))

    multed_pic = pic_tensor * weights_expanded
    
    mixed_pic = torch.sum(multed_pic, dim=0)

    # mixed_pic_gama = torch.pow(mixed_pic, 1/2.2)

    bound = 0
    eps=1e-8
    mixed_pic_gama = ((mixed_pic-bound) / (1.0-2.0*bound)).clamp_min(eps)  ** (1/2.2)

    return mixed_pic_gama