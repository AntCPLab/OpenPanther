# Copyright 2021 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import os, json, logging, argparse
from functools import partial
from typing import Any, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import sys

import flax.linen as nn
from flax.linen.linear import Array

from contextlib import contextmanager

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd
import spu.utils.simulation as ppsim

import sys
print(sys.version)

parser = argparse.ArgumentParser(description="distributed driver.")
parser.add_argument("-c", "--config", default="examples/python/conf/2pc.json")
parser.add_argument("-d", "--dir", default="ppdump")
args = parser.parse_args()

with open(args.config, "r") as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

dump_path = os.path.join(os.path.expanduser("~"), args.dir)

copts = spu_pb2.CompilerOptions()
copts.enable_optimize_denominator_with_broadcast = True

def custom_fn(x: Array) -> Array:
    # print(x.shape)
    # z = intrinsic.f_batch_3_cmp(x)
    b0 = intrinsic.f_less_minus_4(x) # x < -5. 
    b1 = intrinsic.f_less_minus_2(x) # x < -1.5
    z = intrinsic.f_greater_4(x)
    # print(z.shape)
    # z = intrinsic.f_1702_sigmoid(x)
    # b0 = x < -4. 
    # b1 = x < -2.
    # z = x > 4.
    # return z
    return z,b0,b1
@contextmanager
def custom(msg: str, use_onehot: bool = True, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_fn = jnn.gelu
    nn_fn = nn.gelu
    jnn.gelu = partial(custom_fn)
    nn.gelu = partial(custom_fn)
    yield
    # recover back
    jnn.gelu = jnn_fn
    nn.gelu = nn_fn

class Model(nn.Module):
    def __init__(self):
        self.__name__ = "GELU"
        self.act = jnn.gelu

    def __call__(self, x):
        return self.act(x)

def run_on_spu(x):
    with custom("gelu"):
        m = Model()

    x = ppd.device("P1")(lambda x: x)(x)
    z = ppd.device("SPU")(m, copts=copts)(x)
    z = ppd.get(z)
    return z 

def run_on_cpu(x):
    print(x)
    b0 = x < -5
    b1 = x < -1.5
    b2 = x < 5
    return np.concatenate((b0.T,b1.T,b2.T),axis = 1).reshape((1,-1))

if __name__ == "__main__":
    print(sys.version)
    x = (np.random.rand(1,10000) * 10000 - 50) 
    z_cpu = run_on_cpu(x)
    z_spu = run_on_spu(x)
    # print(z_cpu)
    # print((z_cpu == z_spu).all());
    # cmp = (z_cpu == z_spu)
    # # print(z_spu.shape)
    # # print(cmp.shape)
    # count = 0
    # for i in cmp[0]:
    #     if i == False:
    #         print(x[0][(count//3)],z_cpu[0][count], z_spu[0][count]);
    #     count = count + 1