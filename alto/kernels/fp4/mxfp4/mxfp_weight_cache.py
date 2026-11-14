"""Cache of quantized MXFP4 weights within one training step.

Weights change only in the optimizer step, while one step runs several micro-batches (gradient accumulation),
so the weight quantization of the forward pass can be reused. The cache is off unless the training loop
brackets every step with `new_step()` (before the first micro-batch) and `end_step()` (after the optimizer
step), the same contract as the emulation's weight cache (megatron/core/mxfp4_emulation.py: new_step /
end_step). Evaluation, tests and anything else therefore always quantize afresh.

Keys use the weight's storage pointer, shape, dtype, version counter and the quantization settings. The
version counter is an extra guard against in-place updates inside a step; it does not see updates made
through `.data` (as optimizers often do), which is why the cache must be cleared by `end_step()`.
"""
_CACHE = {}
_ENABLED = False


def new_step():
    """Start a training step: clear and enable the cache."""
    global _ENABLED
    _CACHE.clear()
    _ENABLED = True


def end_step():
    """After the optimizer step: clear and disable the cache."""
    global _ENABLED
    _CACHE.clear()
    _ENABLED = False


def enabled() -> bool:
    return _ENABLED


def cached_quantize(weight, settings: tuple, make):
    """make() -> (data, scales); cached for this step if the cache is enabled."""
    if not _ENABLED:
        return make()
    key = (weight.data_ptr(), tuple(weight.shape), weight.dtype, weight._version, settings)
    out = _CACHE.get(key)
    if out is None:
        out = make()
        _CACHE[key] = out
    return out
