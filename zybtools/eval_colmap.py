import numpy as np
from PIL import Image
import os
import struct
import cv2
import sys
import torch
import matplotlib.pyplot as plt

current_script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, parent_dir)

from utils.loss_utils import l1_loss,ssim
from utils.image_utils import psnr, render_net_image

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def bin2depth(depth_map_path, output_path,gt_depth_path):
    depth_map = read_array(depth_map_path)

    per_error_range = [5,10,20,30,50]
    per_error_dict = {item: 0 for item in per_error_range}


    gt_depth_map = cv2.imread(gt_depth_path,cv2.IMREAD_UNCHANGED)
    if gt_depth_map.shape[0]==480:
        scale = 5000
    else:
        scale = 6553.5
    
    gt_depth_map_scaled = gt_depth_map / scale

    depth_map[gt_depth_map_scaled == 0] = 0
    # l1 = l1_loss(torch.tensor(depth_map).to(torch.float64), torch.tensor(gt_depth_map_scaled).to(torch.float64))
    # ssim_out = ssim(torch.tensor(depth_map).to(torch.float64).unsqueeze(0) , torch.tensor(gt_depth_map_scaled).to(torch.float64).unsqueeze(0))
    
    gt_depth_map_scaled[depth_map == 0] = 0

    count = np.sum(np.array(depth_map) > 0.00)
    for per_error in per_error_dict.keys():
        count_notok = np.sum(np.logical_and(((np.abs(np.array(depth_map) - np.array(gt_depth_map_scaled))) > per_error / 1000),np.array(depth_map) > 0.00))
        ok_per = (count - count_notok) / count
        per_error_dict[per_error] += ok_per

    # l1 = l1_loss(torch.tensor(depth_map).to(torch.float64), torch.tensor(gt_depth_map_scaled).to(torch.float64))
    l1mat = np.abs((depth_map - gt_depth_map_scaled))
    masked_l1mat = np.ma.masked_equal(l1mat, 0)

    l1 = np.ma.mean(masked_l1mat)

    # ssim_out = ssim(torch.tensor(depth_map).to(torch.float64).unsqueeze(0) , torch.tensor(gt_depth_map_scaled).to(torch.float64).unsqueeze(0))
    ssim_out = 0
# print(per_error_dict)
    return per_error_dict , l1 , ssim_out
    # min_depth, max_depth = np.percentile(depth_map[depth_map > 0], [2, 98])
    # depth_map[depth_map <= 0] = np.nan
    # depth_map[depth_map < min_depth] = min_depth
    # depth_map[depth_map > max_depth] = max_depth

    # # Normalize to 0-255
    # depth_map = (depth_map - min_depth) / (max_depth - min_depth) * 255
    # depth_map = np.nan_to_num(depth_map).astype(np.uint8)

    # image = Image.fromarray(depth_map).convert('L')
    # image.save(output_path)

# 示例：将所有深度图转换为 PNG 格式
depthmaps_dir = "data/tum/rgbd_dataset_freiburg1_floor_colmap/dense/stereo/depth_maps"
output_dir = "data/tum/rgbd_dataset_freiburg1_floor_colmap/dense/stereo/depth_maps_png"

gt_dir = "data/tum/rgbd_dataset_freiburg1_floor_colmap/depth_colmap"
os.makedirs(output_dir, exist_ok=True)
depth_files = sorted(os.listdir(gt_dir),key=lambda item: int(item.split("_")[0]))

l1_all = 0
ssim_all = 0
for file in sorted(os.listdir(depthmaps_dir),key=lambda item: int(item.split(".")[0])):
    parts = file.split('.')
    depth_name = parts[0] + "_depth.png"
    if "geometric" not in file:
        continue
    print("operating",file.split(".")[0])
    per_error_dicts = []
    if file.endswith(".bin"):
        bin_path = os.path.join(depthmaps_dir, file)
        png_path = os.path.join(output_dir, file.replace(".bin", ".png"))
        gt_depth_path = os.path.join(gt_dir , depth_name)
        
        per_error_dict,l1 , ssim = bin2depth(bin_path, png_path,gt_depth_path)
        per_error_dicts.append(per_error_dict)

        l1_all += l1



sum_dict = {}
count_dict = {}
# 遍历每个字典
for d in per_error_dicts:
    for key, value in d.items():
        # 如果键不存在，初始化为0
        if key not in sum_dict:
            sum_dict[key] = 0
            count_dict[key] = 0
        # 累加值和计数
        sum_dict[key] += value
        count_dict[key] += 1

# 计算平均值
average_per_error_dict = {key: sum_dict[key] / count_dict[key] for key in sum_dict}

l1_all = l1_all / depth_files.__len__()
print(average_per_error_dict)
print("l1:",l1_all * 1000)