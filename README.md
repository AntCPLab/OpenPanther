# PANTHER: Private Approximate Nearest Neighbor Search in the Single Server Setting

This repository contains the source code for our work accepted at **CCS 2025**, titled:
> **[Panther: Private Approximate Nearest Neighbor Search in the Single Server Setting](https://eprint.iacr.org/2024/1774)**


 The core source code for Panther is located in the `experimental/panther` directory.
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

## Requirements

Our implementation is based on the [SPU](https://github.com/secretflow/spu) library, specifically on [this commit](https://github.com/secretflow/spu/commit/94bd4b91cee598003ad2c297def62507b78aa01f), which depends on [bazel](https://bazel.build) for building the project.  We have also integrated two additional libraries:

- **Garbled Circuits** from [emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc).
- **Multi-quiry Seal-PIR** from [PSI](https://github.com/secretflow/psi)

The setup follows the official [build prerequisites for SPU on Linux](https://github.com/secretflow/spu/blob/main/CONTRIBUTING.md#build) :
```
Install gcc>=11.2, cmake>=3.22, ninja, nasm>=2.15, python>=3.9, bazelisk, xxd, lld, numpy
```

We provide the necessary Bazel commands to build and run Panther in the sections below.
These commands have been tested on Ubuntu 22.04 (Linux)

**How to use bazel:**

We prefer to use [`bazelisk`](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file) to install bazel. 

- **Linux**: You can download Bazelisk binary on its [Releases](https://github.com/bazelbuild/bazelisk/releases) page and add it to your `PATH` manually.
   (e.g. copy it to `/usr/local/bin/bazel`).
- Navigate to the `***/OpenPanther` directory and run:
```bash 
bazel --version 
```


Then you can run `cd OpenPanther` and  use `bazel` to build Panther unit test  and e2e test.

⚠️ **Note**: We do not recommend using `bazel build //...` or `bazel build ...` to build all targets. SPU is a large, feature-rich library, and building everything will compile many irrelevant components — this process may take a long time.

**Troubleshooting: Ninja Not Found**

If you encounter the following error:

```shell 
CMake Error: CMake was unable to find a build program corresponding to "Ninja".
``` 

Please install **ninja** using:

```shell
sudo apt install ninja-build
```

<!-- We only need to build the backend of SPU. It can reduce the dependencies of the implentation.   -->
## Code Structure

This section describes the structure of the **Panther** module, located at `experimental/panther`. The key components are:

- `protocols/`: Implementations of core cryptographic subprotocols:
   - `protocols/customize_pir/`:  Customized multi-query PIR protocol 
   - `protocols/*.cc`: Implementations of other subprotocols (top-$k$, truncation, distance computation, etc.) 

- `demo/`: End-to-end demonstrations of Panther on datasets: Deep10M and SIFT(1M).

- `dataset/`: The cluster information from the KMeans model and the data required for KNN queries.

- `benchmark/`: Test files for hybrid building blocks implemented by combining multiple cryptographic subprotocols. Includes randomized end-to-end (e2e) tests.


- `k-means/`: $k$-means training, accuracy evaluation, and script for converting the model into the required input format.


- `BUILD.bazel`: Bazel build configuration file for compiling the Panther framework. 

- `throttle.sh`: Script for network bandwidth throttling, used to simulate different network conditions during performance evaluation.




## End-to-end Test
We provide a [random data version](#random-data-version)  and [real data version](#real-data-version) for end-to-end evaluation. **The random version is only used in performance test, it lets the user quickly reproduce the performance result without downloading the dataset or $k$-means model**.


⚠️ **Note**: 

- A 1M dataset requires 64 GB of memory.
- A 10M dataset requires 256 GB of memory.
- If resources are limited, consider running unit tests.

### Random Data Version
----------------------
#### Execute Random Version  
``` bash
# Sift client
bazel run -c opt //experimental/panther:random_panther_client  --copt=-DTEST_SIFT
# Sift server
bazel run -c opt //experimental/panther:random_panther_server --copt=-DTEST_SIFT

# Amazon client
bazel run -c opt //experimental/panther:random_panther_client  --copt=-DTEST_AMAZON
# Amazon server
bazel run -c opt //experimental/panther:random_panther_server --copt=-DTEST_AMAZON

# 
# Deep1M client
bazel run -c opt //experimental/panther:random_panther_client  --copt=-DTEST_DEEP1M
# Deep1M server
bazel run -c opt //experimental/panther:random_panther_server --copt=-DTEST_DEEP1M

```

#### Execute Random deep1B(10M) Version  
``` bash
# client
bazel run -c opt //experimental/panther:random_panther_client_10M
# server
bazel run -c opt //experimental/panther:random_panther_server_10M
```


### Real Data Version
----------
If you want to run the real data version:

[Step 1]((#datasets).). Download the target dataset 

[Step 2](#sec-kmeans). Obtain the $k$-means model and use `convert_model_to_input.py` to convert the `sift.pth` or `deep10M.pth` to panther  input format. The output will be saved in `experimental/panther/dataset` 

-  **Recommended**:  Using the pretrained $k$-means model , which involves the clustering information and centroids information. 

- Train the $k$-means model by yourself. We provide the train code here. You need to manually tune the Panther parameters based on the model. 

[Step 3](#build--run-demo). Build and run the Panther code

### Dataset
We provide two datasets, which are sourced from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks/):
| Dataset                                                           | Dimensions | Train size | Test size | Neighbors | Distance  | Download                                                                   |
| ----------------------------------------------------------------- | ---------: | ---------: | --------: | --------: | --------- | -------------------------------------------------------------------------- |
| [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)              |         96 |  9,990,000 |    10,000 |       100 | Angular   | [HDF5](http://ann-benchmarks.com/deep-image-96-angular.hdf5) (3.6GB)
| [SIFT](http://corpus-texmex.irisa.fr/)                           |        128 |  1,000,000 |    10,000 |       100 | Euclidean | [HDF5](http://ann-benchmarks.com/sift-128-euclidean.hdf5) (501MB)          |

⚠️ **Note**: The original deep1B provides neighbors based on angular distance. In our code, we have recomputed the neighbors under Euclidean distance using a linear scan.
<a id="sec-kmeans"></a>
### $k$-means Algorithm
We have reproduced the k-means clustering algorithm from [SANNS](), where the implementation of the k-means clustering component relies on the [FAISS](https://github.com/facebookresearch/faiss) library, a commonly used library in ANNS.
#### Dependencies
We provides the dependencies for dataset processing and plaintext 
k-means algorithm here.   
``` bash
conda create -n panther python=3.12.2
conda activate panther
conda install numpy==1.26.4
conda install h5py==3.14.0
conda install pytorch==2.3.0
conda install pytorch::faiss-cpu
```
We recommend using these versions to ensure compatibility and reproducibility, using conda.

 Train model (Not required)
```bash
# <dataset>: sift or deep10M
python3 ./experimental/panther/k-means/sanns_kmeans.py <dataset> 
```
Test model accuracy (Not required)

```bash
# <dataset>: sift or deep10M
python3 ./experimental/panther/k-means/accuracy_test.py <dataset> 
```

#### Convert model to input format (Required)
Make sure that the `/experimental/panther/dataset` directory contains `*.pth` and `*.hdf5` files.

```bash
# <dataset>: sift or deep10M
python3 ./experimental/panther/k-means/convert_model_to_input.py <dataset> 
```

For the SIFT dataset, the output directory structure will be:

```
/experimental/panther/dataset/
├── sift/
│   ├── sift.pth
│   ├── sift-128-euclidean.hdf5
│   ├── sift_dataset.txt
│   ├── sift_test.txt
│   ├── sift_centroids.txt
│   ├── sift_neighbors.txt
│   ├── sift_ptoc.txt
│   └── sift_stash.txt

```

### Build & Run demo 

For Sift:
``` bash
# build the sift demo
# client
bazel build -c opt //experimental/panther:panther_client
# server
bazel build -c opt //experimental/panther:panther_server

# Run the sift demo
# client
bazel run //experimental/panther:panther_client
# server
bazel run //experimental/panther:panther_server
```

For Deep1B_10M:

``` bash
# build the deep1B demo
# client
bazel build -c opt //experimental/panther:panther_client_deep10M
# server
bazel build -c opt //experimental/panther:panther_server_deep10M

# Run the deep1B demo
# client
bazel run //experimental/panther:panther_client_deep10M
# server
bazel run //experimental/panther:panther_server_deep10M
```

## Unit Test
Our framework integrates multiple cryptographic subprotocols. We provide unit tests to verify the correctness of each component in isolation. We hope these subprotocols can be easily reused in other work.

In cases where the evaluator has **constrained computational resources**, it is feasible to test each subprotocol separately. The complete framework is composed of multiple subprotocols, connected via lightweight local computation for intermediate data processing.

**Run Unit Test:**

```bash
# Running Distance Computation Tests
bazel run -c opt //experimental/panther:dist_cmp_test
bazel run -c opt //experimental/panther:dist_cmp_ss_test.
```

```bash
# Running Custom Multi-query PIR Test
bazel run -c opt //experimental/panther/protocol/customize_pir:seal_mpir_test
```

```bash
# Running SS-based Argmax Test
bazel run -c opt //experimental/panther:batch_min_test

# Running Bitwidth Adjustment Test
bazel run -c opt //experimental/panther:bitwidth_adjust_test
``` 
```bash
# Running GC-based Top-K Test
# follow emp-toolkit style
# requires launching server and client separately
# server
bazel run //experimental/panther:topk_test 1 1111 
# client
bazel run //experimental/panther:topk_test 2 1111
```

``` bash
# Running Mixed (SS and GC) Top-k Test
# requires launching server and client separately
# server
bazel run -c opt //experimental/panther:topk_benchmark -- -rank=1 
# client
bazel run -c opt //experimental/panther:topk_benchmark -- -rank=0
```

``` bash 
# Running Running Distance Computation Tests with Truncation
# requires launching server and client separately
# server
azel run -c opt //experimental/panther:distance_benchmark -- -rank=1
# client
azel run -c opt //experimental/panther:distance_benchmark -- -rank=0
```
