from torch_kdtree import build_kd_tree
import torch
from scipy.spatial import KDTree #Reference implementation
import numpy as np

#Dimensionality of the points and KD-Tree
d = 3

#Specify the device on which we will operate
#Currently only one GPU is supported
device = torch.device("cuda")

#Create some random point clouds
points_ref = torch.randn(size=(1000, d), dtype=torch.float32, device=device, requires_grad=True) 
points_ref_100 = points_ref 
points_query = torch.randn(size=(1000, d), dtype=torch.float32, device=device, requires_grad=True)
points_query_100 = points_query 

#Create the KD-Tree on the GPU and the reference implementation
torch_kdtree = build_kd_tree(points_ref_100)
# kdtree = KDTree(points_ref.detach().cpu().numpy())

#Search for the 5 nearest neighbors of each point in points_query
k = 10
dists, inds = torch_kdtree.query(points_query_100, nr_nns_searches=k)
# dists_ref, inds_ref = kdtree.query(points_query.detach().cpu().numpy(), k=k)

#Test for correctness 
#Note that the cupy_kdtree distances are squared
# assert(np.all(inds.cpu().numpy() == inds_ref))
# assert(np.allclose(torch.sqrt(dists).detach().cpu().numpy(), dists_ref, atol=1e-5))

loss = (0.5 * torch.sum(dists))
loss.backward()
grad = points_query.grad 
grad_comp = torch.sum((points_query_100[:, None] - points_ref_100[inds]), axis=-2)
print(torch.allclose(points_query.grad, grad_comp)) #Should print True