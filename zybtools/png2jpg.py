import os
from PIL import Image

# 设置输入文件夹和输出文件夹路径
input_folder = "/home/zhaoyibin/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM_2d/dataset/2dgs/replica/scan1/images_png"  # 包含 JPG 文件的文件夹
output_folder = "/home/zhaoyibin/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM_2d/dataset/2dgs/replica/scan1/images"  # 输出 PNG 文件的文件夹

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 检查文件是否是 JPG 格式
    if filename.lower().endswith(".png"):
        # 构建完整的文件路径
        jpg_path = os.path.join(input_folder, filename)
        
        # 打开 JPG 图片
        with Image.open(jpg_path) as img:
            # 转换为 RGB 模式（确保兼容性）
            img = img.convert("RGB")
            
            # 构建输出文件名（将 JPG 替换为 PNG）
            output_filename = os.path.splitext(filename)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # 保存为 PNG 格式
            img.save(output_path)
            print(f"{output_path} saved")

print("批量转换完成！")