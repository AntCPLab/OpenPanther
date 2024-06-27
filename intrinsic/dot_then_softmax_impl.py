__all__ = ["dot_then_softmax"]

from functools import partial

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def dot_then_softmax(input):
    # Add necessary preprocessing code
    return _dot_then_softmax_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _dot_then_softmax_abstract(input):
    #shape = jax.eval_shape(lambda w, x: jnp.dot(w, x), weight, input).shape  # no FLOPs performed
    shape = input.shape
    dtype = dtypes.canonicalize_dtype(input.dtype)
    ret = ShapedArray(shape, dtype)
    return ret


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _dot_then_softmax_lowering(ctx, input):
    ## FIXME(lwj): needs dot shape, not mul shape
    dtype = mlir.ir.RankedTensorType(input.type)
    return custom_call(
        "dot_then_softmax",
        # Output types
        out_types=[dtype],
        # The inputs:
        operands=[input],
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _dot_then_softmax_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _dot_then_softmax_batch(args, axes):
    print("_dot_then_batch")
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_dot_then_softmax_prim = core.Primitive("dot_then_softmax")
# Change this to True if there are more than 1 output
_dot_then_softmax_prim.multiple_results = False
_dot_then_softmax_prim.def_impl(partial(xla.apply_primitive, _dot_then_softmax_prim))
_dot_then_softmax_prim.def_abstract_eval(_dot_then_softmax_abstract)

mlir.register_lowering(_dot_then_softmax_prim, _dot_then_softmax_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_dot_then_softmax_prim] = _dot_then_softmax_jvp
batching.primitive_batchers[_dot_then_softmax_prim] = _dot_then_softmax_batch
