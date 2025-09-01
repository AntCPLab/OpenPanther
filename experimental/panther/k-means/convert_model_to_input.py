# %%
import torch
from torch.utils import data
import numpy as np
import h5py
import os
from time import time
import faiss
import io
import sys

current_dir = os.getcwd()
data_dir = current_dir + "/experimental/panther/dataset/"
dataset = sys.argv[1]
if dataset == "deep10M":
    max_points_per_cluster = 40
if dataset == "sift":
    max_points_per_cluster = 20 

model_dict = torch.load(data_dir+ dataset+".pth")
ids = model_dict['ids']
cluster_ids = model_dict['cluster_idx']
index = model_dict['index']
centroids = model_dict['centroids']


if dataset == "deep10M":
    file_name = "deep-image-96-angular.hdf5"
elif dataset == "sift":
    file_name = "sift-128-euclidean.hdf5"
else:
    raise ValueError(f"Dataset '{dataset}' is not supported.")

file_path = os.path.join(data_dir, file_name)
data = h5py.File(file_path, "r")
test_x = data['test'][:]
train_x = data['train'][:]
if dataset == "deep10M":
    train_x = ((train_x + 1.0) * 127.5 + 0.5).astype(int)
    test_x = ((test_x + 1.0) * 127.5 + 0.5).astype(int)
test_x = torch.from_numpy(test_x)
train_x = torch.from_numpy(train_x)
print("Dataset load done!")

np.savetxt(data_dir+dataset+"_test.txt",test_x,fmt="%d",delimiter=" ")
np.savetxt(data_dir+dataset+"_dataset.txt",train_x,fmt="%d",delimiter=" ")
np.savetxt(data_dir+dataset+"_centroids.txt",centroids, fmt = "%d", delimiter= " ")

d=test_x.shape[1]
print("dim: ",d)
searchs = faiss.IndexFlatL2(d)
searchs.add(train_x)
D,res = searchs.search(test_x, 10)
neighbors = torch.from_numpy(res)
np.savetxt(data_dir+dataset+"_neighbors.txt",neighbors,fmt = "%d", delimiter= " ")

""" Maps cluster IDs to point IDs """
def save_ptoc(cluster_number, max_points_per_cluster, save):
    result = torch.empty((cluster_number, max_points_per_cluster))
    result.fill_(111111111)
    p_ids = ids.reshape(-1).type(torch.int)
    c_ids = cluster_ids.reshape(-1).type(torch.int)
    order = torch.argsort(c_ids)
    p_ids = p_ids[order]
    c_ids = c_ids[order] 
    counts = torch.bincount(c_ids)
    split_p_ids = torch.split(p_ids,counts.tolist(),dim=0)
    for i in range(cluster_number):
        cluster_points = split_p_ids[i]
        result[i, :len(cluster_points)] = cluster_points
    if save == True:
        np.savetxt(data_dir+dataset+"_ptoc.txt",result,fmt="%d",delimiter=" ")
    return result

all_cluster = centroids.shape[0]
num_cluters = sum(index) - index[-1]
stash_size = index[-1]
print("Total:",all_cluster)
print("Num cluster:", num_cluters," Stash size:", stash_size)
ptoc = save_ptoc(num_cluters,max_points_per_cluster,True)

# Save Stash
_, stash_id = searchs.search(centroids[num_cluters:],1)
stash_id = torch.from_numpy(stash_id)
np.savetxt(data_dir+dataset+"_stash.txt", stash_id, fmt="%d", delimiter= " ")

print("centroids:",centroids.shape)
print("stash:",stash_id.shape)
print("dataset:",train_x.shape)
print("test:",test_x.shape)
print("neighbors:",neighbors.shape)
print("ptoc:",ptoc.shape)
print(index)


