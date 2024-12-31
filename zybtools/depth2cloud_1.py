import torch
import numpy as np



def set_downsample_filter(downsample_scale,H,W,camera_parameter):
    # Get sampling idxs
    [fx ,fy ,cx ,cy] = camera_parameter
    sample_interval = downsample_scale
    h_val = sample_interval * torch.arange(0,int(H/sample_interval)+1)
    h_val = h_val-1
    h_val[0] = 0
    h_val = h_val*W
    a, b = torch.meshgrid(h_val, torch.arange(0,W,sample_interval))
    # For tensor indexing, we need tuple
    pick_idxs = ((a+b).flatten(),)
    # Get u, v values
    v, u = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
    u = u.flatten()[pick_idxs]
    v = v.flatten()[pick_idxs]
    
    # Calculate xy values, not multiplied with z_values
    x_pre = (u-cx)/fx # * z_values
    y_pre = (v-cy)/fy # * z_values
    
    return pick_idxs, x_pre, y_pre


def downsample_and_make_pointcloud2(depth_img, rgb_img,camera_parameter):
    
    H,W = rgb_img.shape[0],rgb_img.shape[1]

    downsample_idxs, x_pre, y_pre =set_downsample_filter(10,H,W,camera_parameter)
    
    colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[downsample_idxs]
    z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[downsample_idxs]/5000.0
    zero_filter = torch.where(z_values!=0)
    filter = torch.where(z_values[zero_filter]<=3.0)
    # print(z_values[filter].min())
    # Trackable gaussians (will be used in tracking)
    z_values = z_values[zero_filter]
    x = x_pre[zero_filter] * z_values
    y = y_pre[zero_filter] * z_values
    points = torch.stack([x,y,z_values], dim=-1)
    colors = colors[zero_filter]
    
    # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
    
    return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
