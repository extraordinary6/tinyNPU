"""Numpy golden models for tinyNPU functional units.

All models are bit-accurate with the corresponding SystemVerilog RTL.
Call sites: cocotb testbenches under tb/test_*/.
"""

from __future__ import annotations

import numpy as np


INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def gemm_i8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """INT8 x INT8 -> INT32 matrix multiply. Shapes: A[M,K] @ B[K,N] -> C[M,N]."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    return a.astype(np.int32) @ b.astype(np.int32)


def bias_relu(acc: np.ndarray, bias: np.ndarray | None,
              bias_en: bool, relu_en: bool) -> np.ndarray:
    """Add optional per-N bias then optional ReLU. INT32 in / out."""
    assert acc.dtype == np.int32
    out = acc.copy()
    if bias_en:
        assert bias is not None and bias.dtype == np.int32
        out = out + bias
    if relu_en:
        out = np.maximum(out, 0)
    return out


def requantize(acc: np.ndarray, mult: int, shift: int) -> np.ndarray:
    """TFLite-lite requantize: saturate_i8( (acc * mult + (1<<(shift-1))) >>> shift ).

    - mult : signed 32-bit multiplier
    - shift: 0..31
    - rounding: round-half-up via pre-shift bias
    - saturation: clip to [-128, 127]
    """
    assert acc.dtype == np.int32
    assert -(1 << 31) <= mult <= (1 << 31) - 1
    assert 0 <= shift <= 31

    product = acc.astype(np.int64) * np.int64(mult)
    round_bias = np.int64(1 << (shift - 1)) if shift > 0 else np.int64(0)
    shifted = (product + round_bias) >> shift
    clipped = np.clip(shifted, INT8_MIN, INT8_MAX)
    return clipped.astype(np.int8)


def requantize_per_channel(acc: np.ndarray, mults, shifts) -> np.ndarray:
    """Per-channel TFLite-lite requantize.

    acc : 1-D length LANES, or 2-D [M, LANES].
    mults, shifts : sequences of length LANES.
    """
    assert acc.dtype == np.int32
    mults = np.asarray(mults, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    assert mults.ndim == 1 and shifts.ndim == 1
    assert mults.shape == shifts.shape
    assert ((-(1 << 31) <= mults) & (mults <= (1 << 31) - 1)).all()
    assert ((0 <= shifts) & (shifts <= 31)).all()

    out = np.empty_like(acc, dtype=np.int8)
    if acc.ndim == 1:
        for c, (m, s) in enumerate(zip(mults, shifts)):
            out[c] = requantize(acc[c:c+1], int(m), int(s))[0]
        return out
    for c, (m, s) in enumerate(zip(mults, shifts)):
        out[:, c] = requantize(acc[:, c].astype(np.int32), int(m), int(s))
    return out

