import numpy as np
import torch
import cv2

pt_path = "/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/scan24/depths/0000.pt"
pt_data = torch.load(pt_path)

shape = (1554,1162)
depth = pt_data['depth']
coord = pt_data['coord']

print("end")