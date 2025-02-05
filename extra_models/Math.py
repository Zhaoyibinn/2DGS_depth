import torch

def rotation_vector_to_rotation_matrix(r):
    """
    将旋转向量转换为旋转矩阵
    输入:
        r: (B, 3) 的张量，表示旋转向量
    输出:
        R: (B, 3, 3) 的张量，表示旋转矩阵
    """
    r = r.unsqueeze(0)
    theta = torch.norm(r, dim=1, keepdim=True)  # 计算旋转角度
    u = r / theta  # 单位向量
    u = u.unsqueeze(-1)  # (B, 3, 1)

    # 构造反对称矩阵 K
    K = torch.zeros((r.shape[0], 3, 3), device=r.device)
    K[:, 0, 1] = -u[:, 2, 0]
    K[:, 0, 2] = u[:, 1, 0]
    K[:, 1, 0] = u[:, 2, 0]
    K[:, 1, 2] = -u[:, 0, 0]
    K[:, 2, 0] = -u[:, 1, 0]
    K[:, 2, 1] = u[:, 0, 0]

    # 计算旋转矩阵
    I = torch.eye(3, device=r.device).unsqueeze(0).repeat(r.shape[0], 1, 1)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * torch.bmm(K, K)
    return R.squeeze()

def matrix_to_rodrigues(R):
    R = R.unsqueeze(0)
    cos_theta = (torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    r = torch.zeros(R.shape[0], 3, device=R.device)
    r[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / (2 * sin_theta)
    r[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / (2 * sin_theta)
    r[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / (2 * sin_theta)
    r = r * theta.unsqueeze(-1)
    return r

