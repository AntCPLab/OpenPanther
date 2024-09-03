for i in {1..64}
do
    ./bazel-bin/experimental/ann/fix_pir_customed/seal_pir_test >> mutli.txt&
done