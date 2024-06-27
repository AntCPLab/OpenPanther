#!/usr/bin/env /opt/homebrew/Caskroom/miniconda/base/bin/python

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


import argparse
import json

import jax.nn as jnn
from jax.nn import softmax, gelu, tanh, relu
import jax.numpy as jnp
import numpy as np

import spu.utils.simulation as ppsim
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/2pc.json")
args = parser.parse_args()

# with open(args.config, 'r') as file:
#     conf = json.load(file)
# ppd.init(conf["nodes"], conf["devices"])

if __name__ == "__main__":
    """
    You can modify the code below for debug purpose only.
    Please DONT commit it unless it will cause build break.
    """

    c_config = spu_pb2.RuntimeConfig(protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64)
    c_config.enable_hal_profile = True
    c_config. enable_lower_accuracy_rsqrt = False
    c_config.fxp_exp_mode = 0
    c_config.fxp_exp_iters = 7
    c_config.enable_heurisitc_truncate = True
    c_config.fxp_fraction_bits = 24

    a_config = spu_pb2.RuntimeConfig(protocol=spu_pb2.ProtocolKind.CHEETAH, field=spu_pb2.FieldType.FM64)
    a_config.fxp_exp_mode = 0
    a_config.fxp_exp_iters = 7
    a_config.enable_hal_profile = True
    a_config.enable_heurisitc_truncate = True
    a_config.fxp_fraction_bits = c_config.fxp_fraction_bits

    c_sim = ppsim.Simulator(2, c_config);
    a_sim = ppsim.Simulator(2, a_config);

    copts = spu_pb2.CompilerOptions()
    copts.disable_div_sqrt_rewrite = True
    copts.enable_pretty_print = False
    copts.enable_optimize_denominator_with_broadcast = True
    #copts.pretty_print_dump_dir = '/home/zhengyancheng.zyc/repo/ppu/ppdump'

    def run_spu(x, y):
        return jnp.dot(x, y)

    def run_aby3(x):
        return jnn.softmax(x, axis=1)

    x = np.random.randn(10, 2048)
    y = np.random.randn(2048, 768)
    a_fn = ppsim.sim_jax(c_sim, run_spu, copts=copts)
    z = a_fn(x, y)

    #diff = g - z3
    #print("aby3 ground error max {} min {} mean {}".format(np.max(diff), np.min(diff), np.mean(np.abs(diff))))

if __name__ == '__main0__':
    x = np.random.randn(4, 78)
    y = np.random.randn(78, 120)
    copts = spu_pb2.CompilerOptions()
    copts.disable_div_sqrt_rewrite = True
    copts.enable_pretty_print = False
    copts.enable_optimize_denominator_with_broadcast = True

    x = ppd.device("P1")(lambda x: x)(x)
    y = ppd.device("P2")(lambda y: y)(y)

    z = ppd.device("SPU")(
        lambda x, y: jnp.dot(x, y),
        copts = copts
    )(x, y)
    z = ppd.get(z)
    print(z)
