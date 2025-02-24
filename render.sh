/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_360 -m results/360/360error --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_360 -m results/360/360error_depth_1.0_s3im0.25_pose --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded

/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_room -m results/tum_room/roomerror --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_room -m results/tum_room/roomerror_error_depth1.0_s3im0.25_pose --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded

/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_floor -m results/floor/floorerror --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python render.py -s data/tum/rgbd_dataset_freiburg1_floor -m results/floor/floorerror_depth1.0_s3im0.25_pose --voxel_size -1.0 --sdf_trunc -1.0 --depth_trunc 5.0 --skip_test --unbounded