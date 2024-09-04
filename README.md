# PANTHER: Private Approximate Nearest Neighbor Search in the Single Server Setting

This repo is the implement of Panther. The implementation source code is in `experimental/ann`. The codes are still under heavy developments, and should not be used in any security sensitive product.

## Building Dependencies
Please refer to the spu and emp-tool documents. Our implentation based on this two library. Their are more details about the dependencies in their repositories. 
We only need to build the backend of SPU. It can reduce the dependencies of the implentation. 


## End-to-end test
We provide a random input version for end-to-end evaluation, which only used in performance analysis, it can let the user quickly reproduce the result without downloading the entire database.  We are finding a better way to provide the database to make it easy to use. 


### Build

(The initial compilation may take a long time.)
```
bazel build //experimental/ann:panther_demo
```
### Execute 
```
bazel run  //experimental/ann:panther_demo & 
bazel run  //experimental/ann:panther_demo -- --rank=1 
```

## Benchmark test
We provid adequate unit test in our repo.
```
# Distance Compute
bazel run //experimental/ann:dist_cmp_test
```


```
# Customed Batch PIR
bazel run //experimenta/ann/fix_pir_customed:seal_mpir_test
```

```
# SS-based Min
bazel run //experimental/ann:batch_argmax_test

# Trunc and Extend
bazel run //experimental/ann:bitwidth_change_test
``` 
```
# GC-based top-k
# follow emp-toolkit style
bazel run //experimental/ann:test_topk 1 1111 &
bazel run //experimental/ann:test_topk 2 1111
```
