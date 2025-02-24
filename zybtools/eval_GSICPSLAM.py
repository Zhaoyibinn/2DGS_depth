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

def eval_depth(rendered_path,gt_path):

    depth_map = cv2.imread(rendered_path,cv2.IMREAD_UNCHANGED)


    # depth_map = read_array(depth_map_path)
    # min_depth, max_depth = np.percentile(depth_map[depth_map > 0], [2, 98])
    # depth_map_save = (depth_map - min_depth) / (max_depth - min_depth) * 255
    # depth_map_save = np.nan_to_num(depth_map_save).astype(np.uint8)
    # image = Image.fromarray(depth_map_save).convert('L')
    # image.save(output_path)

    per_error_range = [5,10,20,30,50]
    per_error_dict = {item: 0 for item in per_error_range}


    gt_depth_map = cv2.imread(gt_path,cv2.IMREAD_UNCHANGED)
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
    l1mat_origin = l1mat
    if gt_depth_map_scaled.shape[1]==640:
        clip_the = 0.5
    else:
        clip_the = 0.1
    
    masked_l1mat = np.ma.masked_equal(l1mat, 0)
    masked_l1mat = np.clip(masked_l1mat,0,clip_the)

    l1 = np.ma.mean(masked_l1mat)

    # ssim_out = ssim(torch.tensor(depth_map).to(torch.float64).unsqueeze(0) , torch.tensor(gt_depth_map_scaled).to(torch.float64).unsqueeze(0))
    ssim_out = 0
# print(per_error_dict)
    return per_error_dict , l1 , ssim_out,l1mat_origin
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
root_path = "data/replica/room0"
gsicpslam_path = "gsicpslam_results/replica/replica_room0"

depthmaps_dir =gsicpslam_path + "/rendered/depth"
txt_save_path = gsicpslam_path + "/depth_l1.txt"
l1_mat_save_path = gsicpslam_path + "/depth_l1_mat_np.npy"


# os.mkdir(l1_mat_save_path)
with open(txt_save_path, 'w', encoding='utf-8') as file:
    # 清空文件内容（'w' 模式会自动清空文件）
    pass  # 这里不需要做任何操作，'w' 模式已经清空了文件

gt_dir = root_path + "/depth_colmap"
# os.makedirs(output_dir, exist_ok=True)
depth_files = sorted(os.listdir(gt_dir),key=lambda item: int(item.split("_")[0]))

depth_render_files = sorted(os.listdir(depthmaps_dir),key=lambda item: int(item.split(".")[0][5:]))

l1_all = 0
ssim_all = 0
per_error_dicts = []
l1mat_origin_all = []

result_save_path = os.path.join(gsicpslam_path,"depth_eval_results.txt")

with open(txt_save_path, 'a', encoding='utf-8') as file:
    for i in range(depth_render_files.__len__()):

        gt_path = os.path.join(gt_dir,depth_files[i])
        rendered_path = os.path.join(depthmaps_dir,depth_render_files[i])

        print("operating:",i)
        per_error_dict,l1 , ssim ,l1mat_origin= eval_depth(rendered_path,gt_path)
        l1mat_origin_all.append(l1mat_origin)
        per_error_dicts.append(per_error_dict)

        l1_all += l1
        
        file.write(str(l1) + "\n")


# l1mat_origin_all_np = np.array(l1mat_origin_all)
# np.save(l1_mat_save_path,l1mat_origin_all_np)
# print("l1 mat np saved to :",l1_mat_save_path)


    # print(l1)



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

with open(result_save_path, 'w', encoding='utf-8') as file:
# 清空文件内容（'w' 模式会自动清空文件）
    pass  # 这里不需要做任何操作，'w' 模式已经清空了文件

with open(result_save_path, 'a', encoding='utf-8') as file:
    # file.write(str(psnr_test_rgb.item()) + "\n")
    file.write(str(l1_all * 1000) + "\n")
    for key in average_per_error_dict:
        file.write(str(average_per_error_dict[key]) + "\n")
