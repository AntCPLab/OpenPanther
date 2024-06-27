__all__ = ["f_mul_no_trunc"]

from functools import partial

from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def f_mul_no_trunc(x):
    # Add necessary preprocessing code
    return _f_mul_no_trunc_prim.bind(x)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _f_mul_no_trunc_abstract(x):
    shape = x.shape
    dtype = dtypes.canonicalize_dtype(x.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _f_mul_no_trunc_lowering(ctx, x):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(x.type)

    return custom_call(
        "f_mul_no_trunc",
        # Output types
        out_types=[dtype],
        # The inputs:
        operands=[x],
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _f_mul_no_trunc_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _f_mul_no_trunc_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_f_mul_no_trunc_prim = core.Primitive("f_mul_no_trunc")
# Change this to True if there are more than 1 output
_f_mul_no_trunc_prim.multiple_results = False
_f_mul_no_trunc_prim.def_impl(partial(xla.apply_primitive, _f_mul_no_trunc_prim))
_f_mul_no_trunc_prim.def_abstract_eval(_f_mul_no_trunc_abstract)

mlir.register_lowering(_f_mul_no_trunc_prim, _f_mul_no_trunc_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_f_mul_no_trunc_prim] = _f_mul_no_trunc_jvp
batching.primitive_batchers[_f_mul_no_trunc_prim] = _f_mul_no_trunc_batch
