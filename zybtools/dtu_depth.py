import torch
import os
import numpy as np
import struct
import cv2

depth_root = "data/scan24/origin_depth"
depth_img_names = os.listdir(depth_root)


def read_pfm(filename):
    """
    读取PFM格式的图像
    """
    with open(filename, 'rb') as f:
        # 读取文件头
        line1 = f.readline().decode('latin-1').strip()
        if line1 == 'PF':
            channels = 3
        elif line1 == 'Pf':
            channels = 1
        else:
            raise Exception('Not a PFM file')
        
        # 读取图像尺寸
        line2 = f.readline().decode('latin-1').strip()
        width, height = map(int, line2.split())
        
        # 读取缩放因子和字节序
        line3 = f.readline().decode('latin-1').strip()
        scale = float(line3)
        bigendian = scale > 0
        scale = abs(scale)
        
        # 读取像素数据
        buffer = f.read()
        fmt = f'{"<>"[bigendian]}{width * height * channels}f'
        decoded = struct.unpack(fmt, buffer)
        
        # 重构图像数据
        shape = (height, width, channels) if channels == 3 else (height, width)
        data = np.flipud(np.reshape(decoded, shape))
        
        return data * scale

if __name__ == "__main__":
    for depth_img_name in depth_img_names:
        depth_img_path = os.path.join(depth_root , depth_img_name)
        if not depth_img_name.endswith(".pfm"):
            continue
        depth = read_pfm(depth_img_path)
        new_width = 1554
        new_height = 1162
        resized_depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        # depth = pt['depth']
        image_idx= int(depth_img_name[-8:-4])
        resized_depth_save_path = os.path.join(depth_root,depth_img_name[-8:-4] + "_depth.tiff")
        cv2.imwrite(resized_depth_save_path,resized_depth)

        print(f"{resized_depth_save_path} saved")
        

    print("end")
