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
file_path = os.path.join(current_dir, "../dataset/deep-image-96-euclidean.hdf5")
deep10M = h5py.File(file_path, "r")
test_x = deep10M['test'][:]
train_x = deep10M['train'][:]
train_x = ((train_x + 1.0) * 127.5 + 0.5).astype(int)
test_x = ((test_x + 1.0) * 127.5 + 0.5).astype(int)
test_x = torch.from_numpy(test_x)
train_x = torch.from_numpy(train_x)
print("Dataset load done!")

# Load k-means cluster
# Recall: 92.639 %
model_dict = torch.load('deep10M.pth')
ids = model_dict['ids'].type(torch.int)
cluster_ids = model_dict['cluster_idx'].type(torch.int)
index = model_dict['index'].type(torch.int)
centroids = model_dict['centroids'].type(torch.int)

# # =====================================
# # fix 10M cluster ics
# check_point = [4661080, 2505289, 1262304, 696089, 387058, 211437, 122682, 65587, 35294]
# offset = 0
# left = 0
# right = check_point[0]
# for i in range(0, len(check_point)):
#     cluster_ids[left:right, :].sub_(offset)
#     left += check_point[i]
#     if i + 1 < len(check_point):
#         right += check_point[i + 1]
#     offset += index[i]
# # =======================================

# parameters
dim = test_x.shape[1]
k = 10
# deep10M
k_cluster = 186

# deep1M
# k_cluster = 113
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
    print(
        "Stash total: ",
        stash_total,
        "Total: ",
        total,
        "Acc: ",
        total / ((i + 1) * k),
    )
print("Recall: ", total / num_test_x * k)  #
