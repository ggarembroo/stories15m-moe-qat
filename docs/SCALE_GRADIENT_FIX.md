# Scale gradient fix

The first QAT implementation used this straight-through estimator.

```python
q = round(w / scale) * scale
return w + (q - w).detach()
```

That passes gradient to `w` but **not** to `scale`. The quantization residual is
detached, and `scale` only appears inside that detached term. A parameter named
"learnable scale" was therefore not learning.

## Fix

```python
q = round(w / scale) * scale
return w + (q - w).detach() + (q - q.detach())
```

Forward value is still `q`. Backward gives

- gradient to `w` through the first residual
- gradient to `scale` through `q - q.detach()`

A minimal check confirms both gradients are non-zero.

## Impact

All runs before 2026-11-23 that claimed "learnable scales" trained only the
weights. The scale values were initialized from calibration but stayed fixed.

The 36M best recipe is being rerun with the fix. Until that finishes, treat the
old 21.78 and 21.88 numbers as "weights trained, scales fixed".

The fix is in `qat.py`. The old local scripts in `hadamard-moe` still have the
old estimator and are not authoritative.
