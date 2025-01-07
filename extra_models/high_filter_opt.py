import torch
import torch.nn.functional as F


def high_filter(img):
    kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
            ], dtype=torch.float32).cuda()
    filtered_image = F.conv2d(img.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))

    return filtered_image.squeeze()