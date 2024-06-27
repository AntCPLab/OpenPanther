__all__ = ["f_less_minus_4"]

from functools import partial

from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
import numpy as np


# Public facing interface
def f_less_minus_4(input):
    # Add necessary preprocessing code
    return _f_less_minus_4_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _f_less_minus_4_abstract(input):
    shape = input.shape
    return ShapedArray(shape, np.dtype(np.bool_))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _f_less_minus_4_lowering(ctx, input):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    inp_shape = mlir.ir.RankedTensorType(input.type).shape
    otype = mlir.ir.RankedTensorType.get(inp_shape, ir.IntegerType.get_signless(1))

    return custom_call(
        "f_less_minus_4",
        # Output types
        out_types=[otype],
        # The inputs:
        operands=[input],
    )

# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _f_less_minus_4_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _f_less_minus_4_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_f_less_minus_4_prim = core.Primitive("f_less_minus_4")
# Change this to True if there are more than 1 output
_f_less_minus_4_prim.multiple_results = False
_f_less_minus_4_prim.def_impl(partial(xla.apply_primitive, _f_less_minus_4_prim))
_f_less_minus_4_prim.def_abstract_eval(_f_less_minus_4_abstract)

mlir.register_lowering(_f_less_minus_4_prim, _f_less_minus_4_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_f_less_minus_4_prim] = _f_less_minus_4_jvp
batching.primitive_batchers[_f_less_minus_4_prim] = _f_less_minus_4_batch
