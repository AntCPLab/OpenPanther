import os
import torch
from torch.utils import data
import numpy as np
import h5py
from time import time
import faiss
import io
import sys

# Reimplement the clustering algorithm in the SANNs paper
# Ref: ”SANNS: Scaling Up Secure Approximate k-Nearest Neighbors Search“
# https://www.usenix.org/conference/usenixsecurity20/presentation/chen-hao

current_dir = os.getcwd()
deep_file_path = os.path.join(current_dir, "deep-image-96-euclidean.hdf5")
deep_h5py = h5py.File(deep_file_path, "r")

train_x = deep_h5py['train'][:]
test_x = deep_h5py['test'][:]

# quantize to 8-bit int
train_x = ((train_x + 1.0) * 127.5 + 0.5).astype(int)
test_x = ((test_x + 1.0) * 127.5 + 0.5).astype(int)
train_x = torch.from_numpy(train_x)
test_x = torch.from_numpy(test_x)

dim = train_x.shape[1]
n_points = train_x.shape[0]

# ==============================
# Save Nearest Neighbors for Test Data
# ==============================
searchs = faiss.IndexFlatL2(dim)
searchs.add(train_x)
D, res = searchs.search(test_x, 10)
res = torch.from_numpy(res)
# np.savetxt("neighbors.txt", res, fmt="%d", delimiter=" ")

# ==============================
# Clustering parameters for deep1M
# kmeans_niters = 1
# max_points_per_cluster = 22
# verbose = True
# frac = 0.56
# stash_size = 25150
# step = 1000
# init_n_c = [55000, 26904, 14793, 8501]
# init_ncentroids = int(init_n_c[0] * 1.1)
# ==============================

# ==============================
# Clustering parameters for deep10M
kmeans_niters = 1
max_points_per_cluster = 40
verbose = True
frac = 0.56
stash_size = 50649
init_n_c = [209727, 107417, 39132, 14424, 5796, 2394]
init_ncentroids = int(init_n_c[0] * 1.1)
step = int(init_ncentroids * 0.1)
# ==============================

# Cumulative offset for cluster index assignment
sum_c = 0
n_cluster = []
centroids = torch.tensor([])
points_ids = torch.tensor([])
cluster_ids = torch.tensor([])

ids = torch.tensor(range(n_points), dtype=torch.int)
train_iter = 0

while n_points > 0:
    # Clustering: Increase ncentroids until oversized clusters are under threshold
    while True:
        original_stdout = sys.stdout
        with io.StringIO() as fake_stdout:
            sys.stdout = fake_stdout
            if init_ncentroids > train_x.shape[0]:
                init_ncentroids = train_x.shape[0]
            kmeans = faiss.Kmeans(
                dim,
                init_ncentroids,
                niter=kmeans_niters,
                verbose=verbose,
                int_centroids=True,
            )
            kmeans.train(train_x)
        sys.stdout = original_stdout

        D, I = kmeans.index.search(train_x, 1)
        labels = torch.from_numpy(I)
        unique_labels, counts = torch.unique(labels, return_counts=True)

        large_point_count = sum(
            size for size in counts if size > max_points_per_cluster
        )
        if large_point_count < frac * n_points:
            break
        else:
            init_ncentroids = init_ncentroids + step

    # Compute the cluster ID to which each point belongs
    D, I = kmeans.index.search(train_x, 1)
    labels = torch.from_numpy(I)
    unique, counts = torch.unique(labels, return_counts=True)

    # Compute the indices of valid clusters
    valid_cluster_indices = torch.nonzero(counts <= max_points_per_cluster).squeeze()

    # Save the number of valid clusters
    n_cluster.append(valid_cluster_indices.shape[0])

    # Extract the centroids of the valid cluster
    valid_centroids = torch.from_numpy(kmeans.centroids)
    valid_centroids = valid_centroids.index_select(dim=0, index=valid_cluster_indices)
    centroids = torch.cat((centroids, valid_centroids), dim=0)
    print("Valid Centrios Shape: ", centroids.shape)

    # Extract the valid points
    saved_train_data_indices = torch.nonzero(
        torch.isin(labels, valid_cluster_indices), as_tuple=True
    )
    saved_train_data = train_x.index_select(dim=0, index=saved_train_data_indices[0])
    saved_ids = ids.index_select(dim=0, index=saved_train_data_indices[0])

    # Find the nearest centrios
    # Compute the new cluster ID to which each point belongs
    centrios_search = faiss.IndexFlatL2(dim)
    centrios_search.add(valid_centroids)
    _, cluster_index = centrios_search.search(saved_train_data, 1)
    cluster_index = torch.from_numpy(cluster_index)
    # offset the new ID
    cluster_index = cluster_index + sum_c

    # uppdate the offset
    sum_c += valid_cluster_indices.shape[0]
    print(sum_c)
    print(cluster_index)
    # save the points id
    points_ids = torch.cat((points_ids, saved_ids), dim=0)

    # save the new cluster ID to which each point belongs
    cluster_ids = torch.cat((cluster_ids, cluster_index), dim=0)

    # save clusters which are larger than the target:
    matching_indices = torch.nonzero(
        ~torch.isin(labels, valid_cluster_indices), as_tuple=True
    )
    train_x = train_x.index_select(dim=0, index=matching_indices[0])
    ids = ids.index_select(dim=0, index=matching_indices[0])
    n_points = train_x.shape[0]

    if n_points < stash_size:
        centroids = torch.cat((centroids, train_x), dim=0)
        n_cluster.append(n_points)
        break

    train_iter = train_iter + 1
    if train_iter < len(init_n_c):
        init_ncentroids = int(init_n_c[train_iter] * 1.1)
        step = int(init_ncentroids * 0.1)
    else:
        init_ncentroids = n_points // max_points_per_cluster + 1

n_cluster = torch.tensor(n_cluster)

tensors_dict = {
    'ids': points_ids,
    'cluster_idx': cluster_ids,
    'index': n_cluster,
    'centroids': centroids,
}
torch.save(tensors_dict, "deep10M.pth")
