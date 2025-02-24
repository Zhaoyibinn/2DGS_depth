import numpy as np
import torch
import cv2

txt_path = "/home/zhaoyibin/3DRE/3DGS/SplaTAM/experiments/TUM/freiburg1_room_seed0/eval/l1.txt"

numbers = []
with open(txt_path, 'r') as file:
    for line in file:
        # 去掉每行的首尾空格，并将字符串转换为数字
        try:
            num = float(line.strip())  # 如果数字是整数，也可以用 int(line.strip())
            numbers.append(num)
        except ValueError:
            print(f"Warning: Line '{line.strip()}' is not a valid number and will be skipped.")

numbers_array = np.array(numbers)



print(numbers_array.mean() * 1000)

for thereshold in [5,10,20,30,50]:
    count_greater_than_threshold = np.sum(numbers_array <(thereshold /1000))
    total_elements = numbers_array.size
    proportion = count_greater_than_threshold / total_elements if total_elements > 0 else 0
    print(thereshold," : ",proportion)


