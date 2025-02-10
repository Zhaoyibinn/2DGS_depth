import torch
from utils.loss_utils import l1_loss,ssim
from utils.image_utils import psnr, render_net_image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def eval_depth(gaussExtractor,TrainCameras):
    results = {}
    rendered_rgbs = gaussExtractor.rgbmaps
    rendered_depths = gaussExtractor.depthmaps
    rendered_image_names = gaussExtractor.image_name

    gt_rgbs = []
    gt_depths = []
    gt_image_names = []
    for camera in TrainCameras:
        
        gt_rgbs.append(torch.clamp(camera.original_image, 0.0, 1.0))
        gt_depths.append(camera.depth)
        gt_image_names.append(camera.image_name)

    image_num = rendered_image_names.__len__()

    if gt_depths[0].shape[1]==640:
        depth_scale = 5000
    else:
        depth_scale = 6553.5

    psnr_test_rgb = 0
    L1_test_depth = 0
    ssim_test_depth = 0
    pbar = tqdm(range(image_num))
    for i in pbar:
        pbar.set_description('Evaling '+str(i))
        rendered_rgb = rendered_rgbs[i]
        # torch.clamp(camera.original_image, 0.0, 1.0)
        gt_rgb = gt_rgbs[i]

        rendered_depth = rendered_depths[i].squeeze()
        gt_depth = gt_depths[i] / depth_scale
        rendered_depth[gt_depth==0]=0


        # fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
        # axes[0].imshow(np.transpose(rendered_rgb, (1, 2, 0)))
        # axes[1].imshow(np.transpose(gt_rgb, (1, 2, 0)))
        
        psnr_test_rgb += psnr(rendered_rgb, gt_rgb).mean().double()
        L1_test_depth += l1_loss(rendered_rgb, gt_rgb)*1000
        ssim_test_depth += ssim(rendered_rgb, gt_rgb)
        # print(psnr(rendered_rgb, gt_rgb).mean().double())
    
    psnr_test_rgb = psnr_test_rgb/image_num
    L1_test_depth = L1_test_depth/image_num
    ssim_test_depth = ssim_test_depth/image_num

    print("psnr_rgb",psnr_test_rgb)
    print("L1_depth",L1_test_depth)
    print("ssim_depth",ssim_test_depth)
    print("---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    return results