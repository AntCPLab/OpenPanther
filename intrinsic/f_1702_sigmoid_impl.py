__all__ = ["f_sigmoid_small"]

from functools import partial

from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def f_1702_sigmoid(input):
    # Add necessary preprocessing code
    return _f_1702_sigmoid_prim.bind(input)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _f_1702_sigmoid_abstract(input):
    shape = input.shape
    dtype = dtypes.canonicalize_dtype(input.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _f_1702_sigmoid_lowering(ctx, input):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(input.type)

    return custom_call(
        "f_1702_sigmoid",
        # Output types
        out_types=[dtype],
        # The inputs:
        operands=[input],
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _f_1702_sigmoid_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _f_1702_sigmoid_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_f_1702_sigmoid_prim = core.Primitive("f_1702_sigmoid")
# Change this to True if there are more than 1 output
_f_1702_sigmoid_prim.multiple_results = False
_f_1702_sigmoid_prim.def_impl(partial(xla.apply_primitive, _f_1702_sigmoid_prim))
_f_1702_sigmoid_prim.def_abstract_eval(_f_1702_sigmoid_abstract)

mlir.register_lowering(_f_1702_sigmoid_prim, _f_1702_sigmoid_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_f_1702_sigmoid_prim] = _f_1702_sigmoid_jvp
batching.primitive_batchers[_f_1702_sigmoid_prim] = _f_1702_sigmoid_batch
