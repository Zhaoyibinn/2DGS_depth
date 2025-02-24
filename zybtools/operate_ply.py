import os
import numpy as np
from plyfile import PlyData,PlyElement
import open3d as o3d
import torch
import math
import torch.nn  as nn




class plyclass:
    def __init__(self,path):
        self.path = path
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

    def cal_sh_color(self,cal_num):

        sh = []
        sh_r = torch.cat((originply._features_dc[cal_num][0][0].reshape([1]),originply._features_rest[cal_num].T[0]),0).tolist()
        sh_g = torch.cat((originply._features_dc[cal_num][0][1].reshape([1]), originply._features_rest[cal_num].T[1]),
                         0).tolist()
        sh_b = torch.cat((originply._features_dc[cal_num][0][2].reshape([1]), originply._features_rest[cal_num].T[2]),
                         0).tolist()
        sh = np.array([sh_r,sh_g,sh_b])

        deg = 3
        # 直接是从原点看的颜色

        SH_C0 = 0.28209479177387814

        SH_C1 = 0.4886025119029199

        SH_C2=[1.0925484305920792,
               -1.0925484305920792,
               0.31539156525252005,
               -1.0925484305920792,
               0.5462742152960396
               ]
        SH_C3=[-0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435]

        points = self._xyz[cal_num]

        x = points[0]
        y = points[1]
        z = points[2]
        color = []

        for sh_1color in sh:
            result = SH_C0 * sh_1color[0]

            result = result - SH_C1 * y * sh_1color[1] + SH_C1 * z * sh_1color[2] - SH_C1 * x * sh_1color[3];

            if (deg > 1):
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z
                result = result +SH_C2[0] * xy * sh_1color[4] +SH_C2[1] * yz * sh_1color[5] +SH_C2[2] * (2.0 * zz - xx - yy) * sh_1color[6] +SH_C2[3] * xz * sh_1color[7] +SH_C2[4] * (xx - yy) * sh_1color[8]

                if (deg > 2):

                    result = result +SH_C3[0] * y * (3.0 * xx - yy) * sh_1color[9] +SH_C3[1] * xy * z * sh_1color[10] +SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_1color[11] +SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_1color[12] +SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_1color[13] +SH_C3[5] * z * (xx - yy) * sh_1color[14] +SH_C3[6] * x * (xx - 3.0 * yy) * sh_1color[15];
            result = result + 0.5
            color.append(result.tolist())


        return color




    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def load_ply( self, instance=False):
        '''
        将之前计算好的点云读取进高斯
        Args:
            path: ply文件路径

        Returns:

        '''
        plydata = PlyData.read(self.path)

        max_sh_degree = 0


        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)  # 三维点的坐标
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  # 不透明性
        scale =np.array([np.asarray(plydata.elements[0]["scale_0"]),np.asarray(plydata.elements[0]["scale_1"]),np.asarray(plydata.elements[0]["scale_2"])])
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])





        # 这里是第一层的球谐系数，共四层，16*3=48个系数，剩下的在下面
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))  # 创建存放剩下45个球谐系数
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])  # 赋值

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # scale参数，暂时不知道干嘛的

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        active_sh_degree = max_sh_degree

        points_num = features_dc.shape[0]
        black_idx = []
        for i in range(points_num):
            features_dc_1 = features_dc[i]
            if (features_dc_1).mean()<=-1.7724538:
                black_idx.append(i)
            # if abs(features_dc_1+1.7724538).mean()<=0.05:
            #     black_idx.append(i)

        self._xyz = nn.Parameter(torch.tensor(xyz[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").transpose(1,
                                                                                     2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").requires_grad_(True))
        rots_trans = rots[:, [3,0,1,2]]
        self._rotation = nn.Parameter(torch.tensor(rots_trans[~np.isin(np.arange(len(xyz)),black_idx)], dtype=torch.float, device="cuda").requires_grad_(True))

        return 0




    def save_ply(self,path_trans):

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path_trans)






# 主程序
if __name__ == "__main__":
    import numpy as np

    # 输入文件路径和输出文件路径
    input_file = "/home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth/results/gsicpslam/TUM/desk/point_cloud/iteration_30000/point_cloud_origin.ply"
    output_file = "/home/zhaoyibin/3DRE/3DGS/3dgs/gaussian-splatting_depth/results/gsicpslam/TUM/desk/point_cloud/iteration_30000/point_cloud.ply"

    originply = plyclass(input_file)
    originply.load_ply()
    originply.save_ply(output_file)
