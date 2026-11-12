"""warp_xla.py -- call Warp tile kernels from jax 0.4.28 through XLA's legacy GPU custom call.

jax 0.4.28 has no FFI. Warp <= 1.17.0 still ships a legacy custom-call bridge (warp/_src/jax/custom_call.py,
"JAX 0.4.25 - 0.7.x"), but that one hard-codes block_dim = 256 and gives every output the launch dims as its shape,
so it cannot launch tile kernels (tile_fft needs its own block_dim and outputs shaped like the data). This module
re-implements the same mechanism with:
  - explicit output shapes and dtypes;
  - tiled launches with any block_dim (the launch dims include the trailing block_dim, as wp.launch_tiled does);
  - a per-descriptor cache, so a call only re-points cached array_t structs at XLA's buffers and launches.
Mechanism (as Warp's): a ctypes callback registered with xla_client.register_custom_call_target; the module is
loaded for (device, block_dim) at lowering; the callback launches with wp_cuda_launch_kernel on XLA's stream.
Modelled on warp/_src/jax/custom_call.py, Warp 1.17.0, Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES,
SPDX-License-Identifier: Apache-2.0.

Kernel arguments: inputs first, then outputs, all contiguous Warp arrays. A JAX operand for a wp.array(dtype=vec2f)
of Warp shape S is a float32 array of shape S + (2,) (vec2d: float64). Scalars are not supported.
Use:
    call = warp_call(kernel, out_specs=[((N, N, 2), "float32")], launch_dims=(N, B), block_dim=B)
    (y,) = call(x)                       # inside or outside jax.jit / lax.scan
"""
import ctypes

_PRIM = None
_CALLBACK = None                                                     # keeps the ctypes callback alive
_KERNELS = [None]                                                    # id -> (kernel, block_dim); id 0 unused
_CACHE = {}                                                          # descriptor bytes -> _Launch
_DEVICE = None
STATS = {"callbacks": 0, "cache_builds": 0, "errors": 0, "last_error": ""}
TARGET = "sqg_warp_call"            # str: jaxlib 0.4.28's register_custom_call_target rejects bytes (job 7839932)


class _Launch:
    """Everything a callback needs that does not depend on the buffer addresses."""
    __slots__ = ("arrays", "params", "bounds", "hooks", "block_dim", "grid_stride", "context", "size", "core")


def _device():
    global _DEVICE
    if _DEVICE is None:
        import warp as wp
        _DEVICE = wp.get_cuda_device(0)                              # CUDA_VISIBLE_DEVICES pins one GPU per process
    return _DEVICE


def _build(desc):
    """Parse '<kernel id>|<launch dims>|<arg shapes ;-separated>' and prepare the launch record."""
    import warp as wp
    from warp._src.context import _build_kernel_launch_bounds
    from warp._src.types import array_t, strides_from_shape
    kid_s, dims_s, args_s = desc.decode().split("|")
    kernel, block_dim = _KERNELS[int(kid_s)]
    dims = [int(d) for d in dims_s.split(",")]
    dev = _device()
    module_exec = kernel.module.load(dev, block_dim)                 # cached by Warp after the lowering's load
    hooks = module_exec.get_kernel_hooks(kernel)
    if not hooks.forward:
        raise RuntimeError(f"no forward entry point for {kernel.key}")
    rec = _Launch()
    rec.bounds = _build_kernel_launch_bounds(dims, kernel, block_dim)
    rec.arrays = []
    shapes = args_s.split(";")
    if len(shapes) != len(kernel.adj.args):
        raise RuntimeError(f"{kernel.key}: {len(shapes)} array descriptors for {len(kernel.adj.args)} arguments")
    for s, arg in zip(shapes, kernel.adj.args):
        shape = [int(d) for d in s.split(",")]
        rec.arrays.append(array_t(0, 0, len(shape), shape, strides_from_shape(shape, arg.type.dtype)))
    rec.params = (ctypes.c_void_p * (1 + len(rec.arrays)))()
    rec.params[0] = ctypes.addressof(rec.bounds)
    for i, a in enumerate(rec.arrays):
        rec.params[i + 1] = ctypes.addressof(a)
    rec.hooks = hooks
    rec.block_dim = block_dim
    rec.grid_stride = int(kernel.grid_stride)
    rec.context = dev.context
    rec.size = rec.bounds.size
    rec.core = wp._src.context.runtime.core
    STATS["cache_builds"] += 1
    return rec


def _callback(stream, buffers, opaque, opaque_len):
    """XLA's custom-call entry (API v2 passes a 5th status pointer, ignored as in Warp's bridge). An exception here
    cannot propagate into XLA: it is counted in STATS and the kernel is not launched, so callers must check results."""
    try:
        STATS["callbacks"] += 1
        desc = ctypes.string_at(opaque, opaque_len)
        rec = _CACHE.get(desc)
        if rec is None:
            rec = _CACHE[desc] = _build(desc)
        for i, a in enumerate(rec.arrays):
            a.data = buffers[i]
        h = rec.hooks
        if rec.core.wp_cuda_launch_kernel(rec.context, h.forward, rec.size, 0, rec.block_dim, rec.grid_stride,
                                          h.cluster_dim, h.forward_smem_bytes, rec.params, stream, None):
            raise RuntimeError("wp_cuda_launch_kernel returned an error")
    except Exception as e:                                           # noqa: BLE001 (reported via STATS)
        STATS["errors"] += 1
        STATS["last_error"] = f"{type(e).__name__}: {e}"


def _create_primitive():
    global _PRIM, _CALLBACK
    import jax
    from jax.interpreters import mlir
    from jax.interpreters.mlir import ir
    from jaxlib.hlo_helpers import custom_call
    try:
        _PRIM = jax.extend.core.Primitive("sqg_warp")
    except AttributeError:
        _PRIM = jax.core.Primitive("sqg_warp")
    _PRIM.multiple_results = True

    def impl(*args, **params):
        raise NotImplementedError("sqg_warp runs only under jax.jit (it lowers to a GPU custom call)")
    _PRIM.def_impl(impl)

    def abstract(*args, kid, out_specs, launch_dims, arg_shapes):
        import numpy as np
        return [jax.core.ShapedArray(shape, np.dtype(dt)) for shape, dt in out_specs]
    _PRIM.def_abstract_eval(abstract)

    cfunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_size_t)
    _CALLBACK = cfunc(_callback)
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    capsule = capsule_new(ctypes.cast(_CALLBACK, ctypes.c_void_p).value, b"xla._CUSTOM_CALL_TARGET", None)
    jax.lib.xla_client.register_custom_call_target(TARGET, capsule, platform="gpu")

    ir_types = {"float32": ir.F32Type.get, "float64": ir.F64Type.get}

    def lowering(ctx, *args, kid, out_specs, launch_dims, arg_shapes):
        kernel, block_dim = _KERNELS[kid]
        if not kernel.module.load(_device(), block_dim):            # compile now, not inside the callback
            raise RuntimeError(f"could not load {kernel.key} for block_dim {block_dim}")
        result_types = [ir.RankedTensorType.get(list(shape), ir_types[dt]()) for shape, dt in out_specs]
        layout = lambda shape: list(range(len(shape) - 1, -1, -1))  # noqa: E731 (row-major)
        opnd_shapes = [ir.RankedTensorType(a.type).shape for a in args]
        desc = f"{kid}|{','.join(map(str, launch_dims))}|{';'.join(','.join(map(str, s)) for s in arg_shapes)}"
        return custom_call(TARGET, result_types=result_types, operands=list(args), backend_config=desc.encode(),
                           operand_layouts=[layout(s) for s in opnd_shapes],
                           result_layouts=[layout(s) for s, _ in out_specs]).results
    mlir.register_lowering(_PRIM, lowering, platform="cuda")


def warp_call(kernel, out_specs, launch_dims, block_dim):
    """A jax-callable for `kernel` (inputs first, then len(out_specs) outputs). out_specs: [(jax shape, "float32" |
    "float64"), ...], vec2 components as the trailing axis. launch_dims: the full Warp launch dims, including the
    trailing block_dim for tile kernels (as wp.launch_tiled builds them)."""
    if _PRIM is None:
        _create_primitive()
    kid = len(_KERNELS)
    _KERNELS.append((kernel, block_dim))
    n_out = len(out_specs)
    n_in = len(kernel.adj.args) - n_out
    if n_in < 1:
        raise ValueError(f"{kernel.key}: needs >= 1 input and {n_out} outputs")
    out_specs = tuple((tuple(int(d) for d in s), str(dt)) for s, dt in out_specs)
    launch_dims = tuple(int(d) for d in launch_dims)

    def warp_shape(arg, jax_shape):
        vec = getattr(arg.type.dtype, "_shape_", ())
        if tuple(jax_shape[len(jax_shape) - len(vec):]) != tuple(vec):
            raise ValueError(f"{kernel.key}.{arg.label}: trailing dims {jax_shape} do not match {vec}")
        shape = tuple(jax_shape[:len(jax_shape) - len(vec)])
        if len(shape) != arg.type.ndim:
            raise ValueError(f"{kernel.key}.{arg.label}: Warp ndim {arg.type.ndim}, got shape {shape}")
        return shape

    def call(*args):
        if len(args) != n_in:
            raise ValueError(f"{kernel.key}: {n_in} inputs expected, got {len(args)}")
        shapes = [warp_shape(a, x.shape) for a, x in zip(kernel.adj.args[:n_in], args)]
        shapes += [warp_shape(a, s) for a, (s, _) in zip(kernel.adj.args[n_in:], out_specs)]
        return _PRIM.bind(*args, kid=kid, out_specs=out_specs, launch_dims=launch_dims, arg_shapes=tuple(shapes))
    return call
