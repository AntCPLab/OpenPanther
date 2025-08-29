# PANTHER: Private Approximate Nearest Neighbor Search in the Single Server Setting

This repository contains the source code for our work accepted at **CCS 2025**, titled:
> **[Panther: Private Approximate Nearest Neighbor Search in the Single Server Setting](https://eprint.iacr.org/2024/1774)**

 The core source code for Panther is located in the `experimental/panther` directory.
The codes are still under heavy developments, and **should not** be used in any security sensitive product.

## Requirements

Our implementation is based on the [SPU](https://github.com/secretflow/spu) library, specifically on [this commit](https://github.com/secretflow/spu/commit/94bd4b91cee598003ad2c297def62507b78aa01f), which depends on [bazel](https://bazel.build) for building the project.  We have also integrated two additional libraries:

- **Garbled Circuits** from [emp-sh2pc](https://github.com/emp-toolkit/emp-sh2pc).
- **Multi-quiry Seal-PIR** from [PSI](https://github.com/secretflow/psi)

The setup follows the official [build prerequisites for SPU on Linux](https://github.com/secretflow/spu/blob/main/CONTRIBUTING.md#build) 
```
Install gcc>=11.2, cmake>=3.26, ninja, nasm>=2.15, python>=3.9, bazelisk, xxd, lld
```

**We provide the necessary Bazel commands to build and run Panther in the sections below.**
These commands have been tested on Ubuntu 22.04 (Linux).

⚠️ **Note**: We do not recommend using `bazel build //...` or `bazel build ...` to build all targets. SPU is a large, feature-rich library, and building everything will compile many irrelevant components — this process may take a long time.

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


## Unit Test
Our framework integrates multiple cryptographic subprotocols. We provide unit tests to verify the correctness of each component in isolation. We hope these subprotocols can be easily reused in other work.

In cases where the evaluator has **constrained computational resources**, it is feasible to test each subprotocol separately. The complete framework is composed of multiple subprotocols, connected via lightweight local computation for intermediate data processing.

**Run Unit Test:**

```
# Distance Compute
bazel run //experimental/panther:dist_cmp_test
bazel run //experimental/panther:ss_dist_cmp_test
```

```
# Customed Batch PIR
bazel run //experimenta/panther/fix_pir_customed:seal_mpir_test
```

```
# SS-based Min
bazel run //experimental/panther:batch_argmax_test

# Trunc and Extend
bazel run //experimental/panther:bitwidth_adjust_test
``` 
```
# GC-based top-k
# follow emp-toolkit style
bazel run //experimental/panther:test_topk 1 1111 &
bazel run //experimental/panther:test_topk 2 1111
```



## End-to-end Test
We provide a [random data version](#random-data-version)  and [real data version](#real-data-version) for end-to-end evaluation. The random version is only used in performance test, it lets the user quickly reproduce the performance result without downloading the dataset or $k$-means model. 

### Random Data Version
----------------------
#### Build Random Version 
(The initial compilation may take a long time.)
The default parameters are for the SIFT dataset.
```
bazel build -c opt //experimental/panther:random_panther_client
bazel build -c opt //experimental/panther:random_panther_server
```
#### Execute Random Version  
```
bazel run //experimental/panther:random_panther_client
bazel run //experimental/panther:random_panther_server
```
### Real Data Version
----------
If you want to run the real data version:

   [Step 1]((#datasets).). Download the target dataset 

[Step 2](#sec-kmeans). Obtain the $k$-means model and use `XX.py` to convert the `**.pth` to panther  input format. The output will be saved in `experimental/panther/dataset` 

-  Download the $k$-means model into datasets, which involves the clustering information and centroids information. 

- Train  the $k$-means model by yourself. We provide the train code here.  

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
```
Python  3.12.2
numpy   1.26.4
torch   2.3.0
faiss   1.8.0
```
We recommend using these versions to ensure compatibility and reproducibility.


### Build & Run demo 
For Deep1B_10M:

```
# build the demo
bazel build -c opt //experimental/panther:random_panther_client
bazel build -c opt //experimental/panther:random_panther_server

# Run the demo
bazel run //experimental/panther:random_panther_client
bazel run //experimental/panther:random_panther_server
```
For Sift:
```
# build the demo
bazel build -c opt //experimental/panther:random_panther_client
bazel build -c opt //experimental/panther:random_panther_server

# Run the demo
bazel run //experimental/panther:random_panther_client
bazel run //experimental/panther:random_panther_server
```
