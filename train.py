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
import torch
from random import randint
from utils.loss_utils import l1_loss,ssim

from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np


import rerun as rr
from torch.utils.tensorboard import SummaryWriter
from zybtools.findPointNormals import findPointNormals,cal_rendered_normal_loss
# from zybtools.depth2cloud_1 import downsample_and_make_pointcloud2,downsample_and_make_pointcloud2_torch, make_pointcloud2_torch
import open3d as o3d
from extra_models.curv_opt import curv_loss
from extra_models.high_filter_opt import high_filter
from extra_models.blur_train.blur import Bayes_fit,mix_pic
import torch.optim as optim
import time

import matplotlib.pyplot as plt

from  extra_models.S3IM.s3im import S3IM

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,args):
    first_iter = 0

    depth_scale = 5000

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,shuffle=False)
    gaussians.training_setup(opt)

    S3IM_loss = S3IM()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_Ll1_depth_for_log = 0.0
    ema_s3im_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    rerun_viewer = args.rerun
    if rerun_viewer:
        rr.init("3dgsviewer")
        rr.spawn(connect=False)
        rr.connect()



    
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration,args)
        

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            camera_sel_range = list(range(viewpoint_stack.__len__()))
        
        if not camera_sel_range:
            camera_sel_range = list(range(viewpoint_stack.__len__()))

        camera_sel = camera_sel_range.pop(randint(0, len(camera_sel_range)-1))
        viewpoint_cam = viewpoint_stack[camera_sel]
        
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # viewpoint_cam = viewpoint_stack.pop(0)
        # 原本是随机弹出，现在就弹出第一个
        # viewpoint_cam = [camera for camera in viewpoint_stack if camera.image_name == "519"][0]

        if args.blur:

            
            before_viewpoint_cam = viewpoint_stack[max(camera_sel-1,0)]
            after_viewpoint_cam = viewpoint_stack[min(camera_sel+1,viewpoint_stack.__len__() - 1)]
            # start = time.time()
            Bayes_pose = Bayes_fit([before_viewpoint_cam,viewpoint_cam,after_viewpoint_cam],vis=False)
            cameras_3 = Scene.init_new_cameras(viewpoint_cam,Bayes_pose)
            # print("bayes time=",time.time() - start)
            render_pkg_before = render(cameras_3[0], gaussians, pipe, background)
            render_pkg_current = render(cameras_3[1], gaussians, pipe, background)
            render_pkg_after = render(cameras_3[2], gaussians, pipe, background)
            RGB_before = render_pkg_before["render"]
            RGB_current = render_pkg_current["render"]
            RGB_after = render_pkg_after["render"]
            # RGB_before_np = np.transpose(np.array(RGB_before.cpu().detach()), (1, 2, 0))
            # RGB_current_np = np.transpose(np.array(RGB_current.cpu().detach()), (1, 2, 0))
            # RGB_after_np = np.transpose(np.array(RGB_after.cpu().detach()), (1, 2, 0))
            # combined_image = np.hstack([RGB_before_np,RGB_current_np,RGB_after_np])
            # plt.imshow(combined_image)

            image = mix_pic([RGB_before,RGB_current,RGB_after],gaussians.blur_camera_vector[int(viewpoint_cam.image_name)])
            _, viewspace_point_tensor, visibility_filter, radii = render_pkg_current["render"], render_pkg_current["viewspace_points"], render_pkg_current["visibility_filter"], render_pkg_current["radii"]
            depth = render_pkg_current['surf_depth']
            rend_dist = render_pkg_current["rend_dist"]
            rend_normal  = render_pkg_current['rend_normal']
            surf_normal = render_pkg_current['surf_normal']




            # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # depth = render_pkg['surf_depth']
            # rend_dist = render_pkg["rend_dist"]
            # rend_normal  = render_pkg['rend_normal']
            # surf_normal = render_pkg['surf_normal']





        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            depth = render_pkg['surf_depth']
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']

        gt_image = viewpoint_cam.original_image.cuda()

        gt_depth = viewpoint_cam.depth.cuda()

        scaled_gt_depth = gt_depth / depth_scale
        depth[scaled_gt_depth.unsqueeze(0) == 0]=0

        # depth_highfiltered = high_filter(depth)
        # depth_highfiltered_np = np.array(depth_highfiltered.cpu().detach())
        # plt.imshow(depth_highfiltered_np)
        # plt.show()
        
        if iteration in args.curv_iterations:
            optimizer_pt_curv = optim.SGD([gaussians._xyz], lr=10.00)
            for i in range(500):
                
                optimizer_pt_curv.zero_grad()
                # fx ,fy ,cx ,cy = 517.29999999999995,516.5,318.60000000000002,255.30000000000001
                # points, colors, z_values, trackable_filter = make_pointcloud2_torch(depth, image,[fx ,fy ,cx ,cy])
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(np.array(points.cpu().detach()))
                # o3d.visualization.draw_geometries([pcd])
                points = gaussians._xyz
                lambda_curv = 1.0
                # normal_cross = cal_rendered_normal_loss(points,render_pkg['rend_normal'])
                loss_curv= curv_loss(points,30) * lambda_curv
                
                # if iteration%10 == 0:
                #     print("正在优化curv, loss = ",loss_curv.item())
                loss_curv.backward()
                optimizer_pt_curv.step()
                if rerun_viewer:
                    rr.set_time_sequence("curv_step", i + iteration)
                    rr.log(f"pt/trackable", rr.Points3D(np.array(points.cpu().detach())[::5,:], radii=0.01))
            

            # curv_loss = loss_curv_normal

            # loss = lambda_curv * curv_loss


        points_3d = np.array(gaussians.get_extratrans_xyz(int(viewpoint_cam.image_name)).cpu().detach())

        if rerun_viewer and (int(viewpoint_cam.image_name) == 519 or iteration%100 == 0):
        # if rerun_viewer and iteration%100 == 0:
            rr.set_time_sequence("step", iteration)
            # print("gaussians.extra_trans[519]:",gaussians.extra_trans[519])
            rr.log(f"pt/trackable", rr.Points3D(points_3d[::5,:], radii=0.01))
            show_image = image
            show_image[show_image>1] = 1.00
            rr.log(f"images/trackable/render",rr.Image(np.transpose(show_image.cpu().detach(), (1, 2, 0))))
            rr.log(f"images/trackable/render_gt",rr.Image(np.transpose(gt_image.cpu().detach(), (1, 2, 0))))

            scaled_gt_depth_vis = scaled_gt_depth/scaled_gt_depth.max()
            scaled_gt_depth_vis[scaled_gt_depth_vis>1] = 1
            depth_vis = depth/scaled_gt_depth.max()
            depth_vis[depth_vis>1] = 1
            rr.log(f"images/trackable/render_depth",rr.Image(np.array(depth_vis.cpu().detach())))
            rr.log(f"images/trackable/render_depth_gt",rr.Image(np.array(scaled_gt_depth_vis.cpu().detach())))

        

        Ll1_depth = l1_loss(depth, scaled_gt_depth)
        
        image_faltten = image.reshape(3,-1).T
        gt_image_faltten = gt_image.reshape(3,-1).T

        idx = torch.randperm(image_faltten.shape[0])[:4096]

        image_faltten_sel = image_faltten[idx]
        gt_image_faltten_sel = gt_image_faltten[idx]
        Ls3im = S3IM_loss(image_faltten_sel, gt_image_faltten_sel)
        
        Ll1 = l1_loss(image, gt_image)
        # print("L1_loss: ",Ll1)
            
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + args.lambda_depth * Ll1_depth + args.lambda_s3im * Ls3im
        


        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0





        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_Ll1_depth_for_log = 0.4 * Ll1_depth.item() + 0.6 * ema_Ll1_depth_for_log
            ema_s3im_for_log = 0.4 * Ls3im.item() + 0.6 * ema_s3im_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    # "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                    # "Depth_L1":f"{ema_Ll1_depth_for_log:.{5}f}",
                    # "s3im_L1":f"{ema_s3im_for_log:.{5}f}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_loss', ema_Ll1_depth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/l1_loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/s3im_loss', ema_s3im_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),args)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
            time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", time_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,args):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    if args.blur:
                        bound = 0
                        eps=1e-8
                        image = ((image-bound) / (1.0-2.0*bound)).clamp_min(eps)  ** (1/2.2)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test +=ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test,ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    
    parser.add_argument("--lambda_depth",type=float, default = 0.0)
    parser.add_argument("--lambda_s3im", type=float,default = 0.25)
    parser.add_argument("--curv_iterations", nargs="+", type=int, default=[])


    parser.add_argument("--rerun",action='store_true',default = False)
    parser.add_argument("--tfboard",action='store_true',default = False)
    parser.add_argument("--extra_pose",action='store_true',default = False)
    parser.add_argument("--blur",action='store_true',default = False)

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,args)

    # All done
    print("\nTraining complete.")