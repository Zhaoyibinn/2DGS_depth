#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import copy
import numpy as np
import torch

from utils.graphics_utils import getWorld2View2





class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling



        # jump_idx_num = 100
        # less_cameras= [scene_info.train_cameras[i] for i in range(len(scene_info.train_cameras)) if (i + 1) % jump_idx_num == 0]
        # print("每过",jump_idx_num,"帧才选择View进mesh")

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        train_max = max(int(camera.image_name) for camera in self.train_cameras[1])
        test_max = max(int(camera.image_name) for camera in self.test_cameras[1])
        cameras_idx_max = max(train_max,test_max)
        self.gaussians.cameras_idx_max = cameras_idx_max
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            extra_trans_pth = os.path.join(self.model_path,"point_cloud","iteration_" + str(self.loaded_iter),"extra_trans.pth")
            if os.path.isfile(extra_trans_pth):
                self.gaussians.init_extra_pose()
                self.gaussians.load_extra_pose(extra_trans_pth)
                print(f"找到优化的pose")
            else:
                self.gaussians.init_extra_pose()
                print(f"没有找到优化的pose，单位初始化")
            
        else:
            self.gaussians.create_from_pcd(args,scene_info.point_cloud, self.cameras_extent)

    def init_new_cameras(origin_camera,more_poses):
        after_camera ,before_camera = copy.deepcopy(origin_camera),copy.deepcopy(origin_camera)
        cameras_3 = [before_camera,origin_camera,after_camera]
        more_poses = np.linalg.inv(more_poses)
        R = more_poses[:,:3,:3]
        R = R.transpose(0,2,1)
        t = more_poses[:,:3,3]
        for i in range(3):
            cameras_3[i].world_view_transform = torch.tensor(getWorld2View2(R[i], t[i])).transpose(0, 1).cuda()
            cameras_3[i].full_proj_transform = (cameras_3[i].world_view_transform.unsqueeze(0).bmm(cameras_3[i].projection_matrix.unsqueeze(0))).squeeze(0)
            cameras_3[i].camera_center = cameras_3[i].world_view_transform.inverse()[3, :3]
        
        return cameras_3
    
    def scene2mask(self):
        self.gaussians._features_rest.zero_()

        self.gaussians._features_dc.fill_(0.5/0.28209479177387)


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_extra_trans(os.path.join(point_cloud_path, "extra_trans.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]