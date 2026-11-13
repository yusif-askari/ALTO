"""Low-overhead access to the `torchtitan::*` MXFP4 triton_ops in eager mode.

Each `torch.ops.torchtitan.<op>(...)` call goes through the dispatcher, the custom-op autograd wrapper and
an output-aliasing check before reaching the Python implementation: about 150 us of CPU time per call on
the test system, with 8-12 calls per forward + backward of one MXFP4 layer. That dominated the CPU time of a
layer and left the GPU waiting. In eager mode this module calls the registered Python implementation
directly; under torch.compile it still goes through `torch.ops` so tracing sees the custom ops.

Set ALTO_MXFP4_EAGER_OPS=0 to always use `torch.ops`.
"""
import os

import torch
from torch._library.custom_ops import _maybe_get_opdef

_USE_EAGER = os.environ.get("ALTO_MXFP4_EAGER_OPS", "1") == "1"


class _Ops:
    def __getattr__(self, name):
        op = getattr(torch.ops.torchtitan, name)
        opdef = _maybe_get_opdef(f"torchtitan::{name}") if _USE_EAGER else None
        if opdef is None:
            fn = op
        else:
            impl = opdef._init_fn

            def fn(*args, **kwargs):
                if torch.compiler.is_compiling():
                    return op(*args, **kwargs)
                return impl(*args, **kwargs)

        setattr(self, name, fn)        # cache: __getattr__ is only called on the first access
        return fn


ops = _Ops()
