import torch
from torch.utils import data
import numpy as np
import h5py
import os
from time import time
import faiss
import io
import sys

# Load dataset
current_dir = os.getcwd()
data_dir = current_dir + "/experimental/panther/dataset/"
dataset = sys.argv[1]
if dataset == "deep10M":
    file_name = "deep-image-96-angular.hdf5"
elif dataset == "sift":
    file_name = "sift-128-euclidean.hdf5"
else:
    raise ValueError(f"Dataset '{dataset}' is not supported.")

file_path = os.path.join(data_dir, file_name)
data_h5py = h5py.File(file_path, "r")
test_x = data_h5py['test'][:]
train_x = data_h5py['train'][:]
if dataset == "deep10M":
    train_x = ((train_x + 1.0) * 127.5 + 0.5).astype(int)
    test_x = ((test_x + 1.0) * 127.5 + 0.5).astype(int)
test_x = torch.from_numpy(test_x)
train_x = torch.from_numpy(train_x)
print("Dataset load done!")

# Load k-means cluster
# Recall: 92.639 %
model_dict = torch.load(data_dir+dataset+'.pth')
ids = model_dict['ids'].type(torch.int)
cluster_ids = model_dict['cluster_idx'].type(torch.int)
index = model_dict['index'].type(torch.int)
centroids = model_dict['centroids'].type(torch.int)

# parameters
dim = test_x.shape[1]
k = 10
# deep10M
k
if dataset == "deep10M":
    k_cluster = 186

# deep1M
if dataset == "sift":
    k_cluster = 123

print(model_dict['index']) 
all_cluster = centroids.shape[0]
num_clusters = sum(index) - index[-1]
print("Num_clusters: ", num_clusters)
print("Num_Clusters + Stash: ", all_cluster)

# Search stash point from train datatset to obtain the Stash ID
all_data_search = faiss.IndexFlatL2(dim)
all_data_search.add(train_x)
_, stash_id = all_data_search.search(centroids[num_clusters:], 1)
stash_id = torch.from_numpy(stash_id)
_, neighbor_x = all_data_search.search(test_x, k)

# Search test data from stash
stash_search = faiss.IndexFlatL2(dim)
stash_search.add(centroids[num_clusters:])
_, stash_search_id = stash_search.search(test_x, k)

# Search test data from centrois
c_search = faiss.IndexFlatL2(dim)
c_search.add(centroids[:num_clusters])
_, id_cluster_res = c_search.search(test_x, k_cluster)

num_test_x = test_x.shape[0]
total = 0
stash_total = 0
for i in range(num_test_x):
    query_point = test_x[i : i + 1]
    exp_kann_res = neighbor_x[i]

    # Get the nearest stash points
    stash_search_res = torch.from_numpy(stash_search_id[i])
    stash_point = stash_id.index_select(dim=0, index=stash_search_res)

    # Return the points in the nearest clusters
    cluster_res = torch.from_numpy(id_cluster_res[i])
    candidate_points_idx_mask = torch.nonzero(
        torch.isin(cluster_ids, cluster_res), as_tuple=True
    )[0]
    candidate_points_idx = ids.index_select(
        dim=0, index=candidate_points_idx_mask
    ).type(torch.int)
    candidate_points = train_x.index_select(dim=0, index=candidate_points_idx)

    candidate_search = faiss.IndexFlatL2(dim)
    candidate_search.add(candidate_points)
    _, nearest_point_in_cluster = candidate_search.search(query_point, k)
    nearest_point_in_cluster = torch.from_numpy(nearest_point_in_cluster)
    cmp_id = candidate_points_idx.index_select(dim=0, index=nearest_point_in_cluster[0])

    total += np.intersect1d(cmp_id, exp_kann_res).shape[0]
    stash_total = np.intersect1d(exp_kann_res, stash_point).shape[0]
    total += stash_total
    if i % 100 == 0:
        print(
            "Stash total: ",
            stash_total,
            "Total: ",
            total,
            "Acc: ",
            total / ((i + 1) * k),
        )
print("Recall: ", total / num_test_x * k)  #
