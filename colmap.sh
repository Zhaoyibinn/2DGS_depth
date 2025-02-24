# cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/tum/rgbd_dataset_freiburg1_360
# colmap image_undistorter --image_path images_colmap --input_path sparse/0 --output_path dense
# colmap patch_match_stereo --workspace_path dense
# colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply


# cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/tum/rgbd_dataset_freiburg1_room
# colmap image_undistorter --image_path images_colmap --input_path sparse/0 --output_path dense
# colmap patch_match_stereo --workspace_path dense
# colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply


# cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/replica/office0
# # colmap image_undistorter --image_path images_colmap --input_path sparse/0 --output_path dense
# colmap patch_match_stereo --workspace_path dense
# colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply

# cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/replica/office1
# colmap image_undistorter --image_path images_colmap --input_path sparse/0 --output_path dense
# colmap patch_match_stereo --workspace_path dense
# colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply

cd /home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/data/tum/rgbd_dataset_freiburg1_room
colmap image_undistorter --image_path images_colmap --input_path sparse/0 --output_path dense
colmap patch_match_stereo --workspace_path dense
colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply