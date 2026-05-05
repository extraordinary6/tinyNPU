"""im2col helpers and a pure-numpy Conv2D reference.

Phase 13 of tinyNPU runs Conv2D entirely in software on top of the existing
INT8 GEMM engine: the host first im2col-expands the input feature map and
re-shapes the kernel, then feeds the resulting matrices through tinyNPU_top
exactly like a plain GEMM. The RTL is unchanged.

Layout conventions (host-side):
    ifm     : INT8 array of shape [H, W, Cin]
    kernel  : INT8 array of shape [kh, kw, Cin, Cout]
    A (im2col) : INT8 array of shape [H' * W', kh * kw * Cin]
                 row index = i * W' + j  (i = output row, j = output col)
                 col index = ki * kw * Cin + kj * Cin + ci
    B (kernel reshape) : INT8 array of shape [kh * kw * Cin, Cout]
                 same flatten order over (ki, kj, ci) as A's columns
    out     : INT32 array of shape [H', W', Cout]

H' and W' follow the standard conv arithmetic:
    H' = (H + 2*padding - kh) // stride + 1
    W' = (W + 2*padding - kw) // stride + 1
"""

from __future__ import annotations

import numpy as np


def output_shape(H: int, W: int, kh: int, kw: int,
                 stride: int = 1, padding: int = 0) -> tuple[int, int]:
    """Standard valid-conv output spatial dims with optional zero padding."""
    Hp = (H + 2 * padding - kh) // stride + 1
    Wp = (W + 2 * padding - kw) // stride + 1
    assert Hp > 0 and Wp > 0, (
        f"output dims must be positive: H={H} W={W} kh={kh} kw={kw} "
        f"stride={stride} padding={padding}"
    )
    return Hp, Wp


def im2col(ifm: np.ndarray, kh: int, kw: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """Expand IFM[H,W,Cin] into A[H'*W', kh*kw*Cin] with row-major patches.

    Patches are extracted in (i, j) row-major output order; each patch is
    flattened in (ki, kj, ci) order. dtype is preserved (expect INT8).
    """
    assert ifm.ndim == 3
    H, W, Cin = ifm.shape
    Hp, Wp = output_shape(H, W, kh, kw, stride, padding)

    if padding > 0:
        padded = np.zeros((H + 2 * padding, W + 2 * padding, Cin), dtype=ifm.dtype)
        padded[padding:padding + H, padding:padding + W, :] = ifm
    else:
        padded = ifm

    A = np.empty((Hp * Wp, kh * kw * Cin), dtype=ifm.dtype)
    for i in range(Hp):
        for j in range(Wp):
            patch = padded[i * stride : i * stride + kh,
                           j * stride : j * stride + kw, :]
            A[i * Wp + j, :] = patch.reshape(-1)
    return A


def kernel_to_b(kernel: np.ndarray) -> np.ndarray:
    """Reshape K[kh,kw,Cin,Cout] into B[kh*kw*Cin, Cout] with matching order."""
    assert kernel.ndim == 4
    kh, kw, Cin, Cout = kernel.shape
    return kernel.reshape(kh * kw * Cin, Cout)


def conv2d_via_gemm(ifm: np.ndarray, kernel: np.ndarray,
                    stride: int = 1, padding: int = 0) -> np.ndarray:
    """im2col + INT32 matmul. Returns INT32 [H', W', Cout]."""
    assert ifm.dtype == np.int8 and kernel.dtype == np.int8
    H, W, Cin = ifm.shape
    kh, kw, Cin_k, Cout = kernel.shape
    assert Cin == Cin_k, f"Cin mismatch: ifm={Cin} kernel={Cin_k}"
    Hp, Wp = output_shape(H, W, kh, kw, stride, padding)

    A = im2col(ifm, kh, kw, stride, padding)
    B = kernel_to_b(kernel)
    flat = A.astype(np.int32) @ B.astype(np.int32)
    return flat.reshape(Hp, Wp, Cout)


def conv2d_reference(ifm: np.ndarray, kernel: np.ndarray,
                     stride: int = 1, padding: int = 0) -> np.ndarray:
    """Direct nested-loop conv2d reference (no im2col). Returns INT32 [H',W',Cout].

    Independent implementation used to cross-check conv2d_via_gemm.
    """
    assert ifm.dtype == np.int8 and kernel.dtype == np.int8
    H, W, Cin = ifm.shape
    kh, kw, Cin_k, Cout = kernel.shape
    assert Cin == Cin_k
    Hp, Wp = output_shape(H, W, kh, kw, stride, padding)

    if padding > 0:
        padded = np.zeros((H + 2 * padding, W + 2 * padding, Cin), dtype=np.int32)
        padded[padding:padding + H, padding:padding + W, :] = ifm.astype(np.int32)
    else:
        padded = ifm.astype(np.int32)

    out = np.zeros((Hp, Wp, Cout), dtype=np.int32)
    k_i32 = kernel.astype(np.int32)
    for i in range(Hp):
        for j in range(Wp):
            patch = padded[i * stride : i * stride + kh,
                           j * stride : j * stride + kw, :]
            for co in range(Cout):
                out[i, j, co] = int((patch * k_i32[:, :, :, co]).sum())
    return out
