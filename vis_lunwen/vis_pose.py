import numpy as np 

import matplotlib.pyplot as plt
import cv2
import torch

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

npz_path = "/home/zhaoyibin/3DRE/3DGS/2d-gaussian-splatting/vis/office0/vis_29000.npz"
npz = np.load(npz_path)



gt ,error,better = npz['gt'],npz['error'],npz['better']

gt_t,error_t ,better_t= gt[:,:3,3],error[:,:3,3],better[:,:3,3]


plt.figure(figsize=(12,9))

ate_error,transed_error,_ = evaluate_ate(gt,error)
print("error_ate",ate_error)
ate_better,transed_better,_ = evaluate_ate(gt,better)
print("our_ate",ate_better)






plt.plot(gt_t[:,0],gt_t[:,1],color='orange',label = "GT")
plt.plot(transed_error.T[:,0],transed_error.T[:,1],color='green',label = "Origin Pose")
plt.plot(transed_better.T[:,0],transed_better.T[:,1],color='black',label = "Our Pose")

# plt.plot(gt_t[:,0],gt_t[:,2],color='orange',label = "GT")
# plt.plot(transed_error.T[:,0],transed_error.T[:,2],color='green',label = "Origin Pose")
# plt.plot(transed_better.T[:,0],transed_better.T[:,2],color='black',label = "Our Pose")


plt.legend(fontsize=14,loc='upper right')

plt.show()
print("end")