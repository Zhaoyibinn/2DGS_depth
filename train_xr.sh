# conda init bash
# conda activate 2dgs_kd




/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python train.py -s data/tum/rgbd_dataset_freiburg1_360 --model_path results/360/360error_pose --test_iterations 100 5000 10000 10000 15000 20000 25000 30000 30001 --extra_pose --checkpoint_iterations 30000
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python train.py -s data/tum/rgbd_dataset_freiburg1_desk_back --model_path results/desk/desk_pose --test_iterations 100 5000 10000 10000 15000 20000 25000 30000 30001 --extra_pose --checkpoint_iterations 30000
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python train.py -s data/tum/rgbd_dataset_freiburg1_floor --model_path results/floor/floorerror_pose --test_iterations 100 5000 10000 10000 15000 20000 25000 30000 30001 --extra_pose --checkpoint_iterations 30000
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python train.py -s data/tum/rgbd_dataset_freiburg1_room --model_path results/tum_room/roomerror_pose --test_iterations 100 5000 10000 10000 15000 20000 25000 30000 30001 --extra_pose --checkpoint_iterations 30000
/home/zhaoyibin/anaconda3/envs/2dgs_kd/bin/python train.py -s data/tum/rgbd_dataset_freiburg2_xyz --model_path results/360/360error_pose --test_iterations 100 5000 10000 10000 15000 20000 25000 30000 30001 --extra_pose --checkpoint_iterations 30000
