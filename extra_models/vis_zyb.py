import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import cv2
from PIL import Image
import seaborn as sns
import pandas as pd


def align(model, data):

    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error

def evaluate_ate_without_align(gt_traj, est_traj):

    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    gt_traj_pts_arr = np.array(gt_traj_pts)
    gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
    gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
    est_traj_pts_arr = np.array(est_traj_pts)
    est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
    est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

    alignment_error = gt_traj_pts - est_traj_pts
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0))


    # _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()
    

    return avg_trans_error

def evaluate_ate(gt_traj, est_traj):

    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    gt_traj_pts_arr = np.array(gt_traj_pts)
    gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
    gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
    est_traj_pts_arr = np.array(est_traj_pts)
    est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
    est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

    rot, trans, trans_error = align(gt_traj_pts, est_traj_pts)
    
    transed_est_traj_pts = np.linalg.inv(rot) * est_traj_pts - np.linalg.inv(rot) * trans
    avg_trans_error = trans_error.mean()

    return avg_trans_error,transed_est_traj_pts,trans_error

def vis_pose_error(error_poses_cam,gt_poses_cam,extra_trans = None,logate = False):

    fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(8, 16)) 
    error_poses_np_R = np.array([pose.R  for pose in sorted(error_poses_cam[1.0],key=lambda x: int(x.image_name))])
    gt_poses_np_R = np.array([pose.R  for pose in sorted(gt_poses_cam[1.0],key=lambda x: int(x.image_name))])

    error_poses_np_R = np.linalg.inv(error_poses_np_R)
    gt_poses_np_R = np.linalg.inv(gt_poses_np_R)

    error_poses_np_t = np.array([pose.T  for pose in sorted(error_poses_cam[1.0],key=lambda x: int(x.image_name))])
    gt_poses_np_t = np.array([pose.T  for pose in sorted(gt_poses_cam[1.0],key=lambda x: int(x.image_name))])
    
    pic_num = error_poses_np_R.shape[0]
    
    error_poses_np_T = np.tile(np.eye(4), (pic_num, 1, 1))
    error_poses_np_T[:,:3,:3] = error_poses_np_R
    error_poses_np_T[:,:3,3] = error_poses_np_t
    error_poses_np_T = np.linalg.inv(error_poses_np_T)
    # W2C

    gt_poses_np_T = np.tile(np.eye(4), (pic_num, 1, 1))
    gt_poses_np_T[:,:3,:3] = gt_poses_np_R
    gt_poses_np_T[:,:3,3] = gt_poses_np_t
    gt_poses_np_T = np.linalg.inv(gt_poses_np_T)
    

    ate_error,transed_est_traj_pts,trans_error_origin = evaluate_ate(gt_poses_np_T , error_poses_np_T)




    if logate:
        print(f"ATE origin error:{ate_error}")
    # plt.figure()


    gt_poses_np_t = gt_poses_np_T[:,:3,3]
    x = gt_poses_np_t[:,0]
    y = gt_poses_np_t[:,1]
    z = gt_poses_np_t[:,2]

    ax1.plot(x, y,label ="gt")
    ax2.plot(x, z,label ="gt")

    
    # error_poses_np_t = error_poses_np_T[:,:3,3]
    error_poses_np_t = np.array(transed_est_traj_pts.T)
    x = error_poses_np_t[:,0]
    y = error_poses_np_t[:,1]
    z = error_poses_np_t[:,2]

    ax1.plot(x, y,label ="error")
    ax2.plot(x, z,label ="error")



    # extra_trans默认是高斯的变换，对应到世界就是反变换

    # 这里的R和t都是

    if extra_trans != None:
        extra_poses_np_T = np.array(extra_trans.cpu().detach())

        extra_poses_np_T_T = np.linalg.inv(extra_poses_np_T)
        better_poses_np_T = extra_poses_np_T_T @ error_poses_np_T

        ate_error_better,transed_est_traj_pts,trans_error_better = evaluate_ate(gt_poses_np_T , better_poses_np_T)
        if logate:
            print(f"ATE better error:{ate_error_better}")
        
        # better_poses_np_t = better_poses_np_T[:,:3,3]
        better_poses_np_t = np.array(transed_est_traj_pts.T)
        
        x = better_poses_np_t[:,0]
        y = better_poses_np_t[:,1]
        z = better_poses_np_t[:,2]
        ax1.plot(x, y,label ="extra_trans" )
        ax2.plot(x, z,label ="extra_trans" )

        # print("ok")
    ax1.legend(loc = 'upper right')
    ax2.legend(loc = 'upper right')
    ax1.set_title('xy')
    ax2.set_title('xz')
    plt.savefig('vis/trajectory.png', dpi=300)
    trajectory_img = cv2.imread('vis/trajectory.png')

    # plt.show()
    # fig.canvas.draw()  # 绘制图形
    # fig.show()
    # trajectory_img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


    img_array = np.array(trajectory_img)


    # plt.clf()
    
    # data = pd.DataFrame({
    # 'Error': np.concatenate([trans_error_origin, trans_error_better]),
    # 'Group': ['Origin'] * len(trans_error_origin) + ['Better'] * len(trans_error_better)
    # })
    # sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    # g = sns.FacetGrid(data, row="Group", aspect=9, height=1.2, margin_titles=True)
    # g.map_dataframe(sns.kdeplot, x="Error", fill=True, alpha=0.5, linewidth=1.5)
    # g.map_dataframe(sns.kdeplot, x="Error", color='black')

    # g.fig.subplots_adjust(hspace=-.5)
    # g.set_titles(col_template="{col_name}", row_template="{row_name}", size=12)
    # g.set(yticks=[], xlabel="Error", ylabel="")
    # g.despine(left=True)
    # plt.show()

    
    return img_array