"""Minimal host-side tinyNPU driver example.

This script demonstrates:
1) CSR programming (M/N/K/base/flags/quant params),
2) SRAM tile packing/layout for IFM/W/BIAS/REQ params,
3) start/poll/readback flow.

`HostIO` abstracts bus access so this can run on real hardware, simulation,
or a mock backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np


A_CTRL = 0x004
A_STATUS = 0x008
A_M = 0x00C
A_N = 0x010
A_K = 0x014
A_IFM = 0x018
A_W = 0x01C
A_OFM = 0x020
A_BIAS = 0x024
A_FLAGS = 0x028
A_REQ_MULT = 0x02C
A_REQ_SHIFT = 0x030
A_REQ_MULT_BASE = 0x034
A_REQ_SHIFT_BASE = 0x038


@dataclass
class HostIO:
    apb_write: Callable[[int, int], None]
    apb_read: Callable[[int], int]
    ifm_write: Callable[[int, int], None]
    w_write: Callable[[int, int], None]
    bias_write: Callable[[int, int], None]
    ofm_read: Callable[[int], int]


def _pack_i8_row(row: np.ndarray) -> int:
    val = 0
    for i, v in enumerate(row):
        val |= (int(v) & 0xFF) << (8 * i)
    return val


def _pack_i8_tile(tile: np.ndarray, rows: int, cols: int) -> int:
    val = 0
    for r in range(rows):
        for c in range(cols):
            val |= (int(tile[r, c]) & 0xFF) << (8 * (r * cols + c))
    return val


def _pack_i32_lanes(values: List[int]) -> int:
    val = 0
    for i, v in enumerate(values):
        val |= (int(v) & 0xFFFF_FFFF) << (32 * i)
    return val


def _unpack_i8_row(word: int, cols: int) -> List[int]:
    out = []
    for c in range(cols):
        b = (word >> (8 * c)) & 0xFF
        out.append(b - 256 if b & 0x80 else b)
    return out


class TinyNPUHost:
    def __init__(self, io: HostIO, rows: int = 4, cols: int = 4):
        self.io = io
        self.rows = rows
        self.cols = cols

    def load_ifm(self, A: np.ndarray, ifm_base: int):
        M, K = A.shape
        assert K % self.cols == 0
        k_tiles = K // self.cols
        for k in range(k_tiles):
            a_slice = A[:, k * self.cols : (k + 1) * self.cols]
            for i in range(M):
                self.io.ifm_write(ifm_base + k * M + i, _pack_i8_row(a_slice[i]))

    def load_w(self, W: np.ndarray, w_base: int):
        K, N = W.shape
        assert K % self.rows == 0 and N % self.cols == 0
        k_tiles = K // self.rows
        n_tiles = N // self.cols
        for n in range(n_tiles):
            for k in range(k_tiles):
                slab = W[k * self.rows : (k + 1) * self.rows, n * self.cols : (n + 1) * self.cols]
                self.io.w_write(w_base + n * k_tiles + k, _pack_i8_tile(slab, self.rows, self.cols))

    def configure_and_start(
        self,
        *,
        M: int,
        N: int,
        K: int,
        ifm_base: int,
        w_base: int,
        ofm_base: int,
        flags: int = 0,
        req_mult: int = 1,
        req_shift: int = 0,
        bias_base: int = 0,
        req_mult_base: int = 0,
        req_shift_base: int = 0,
    ):
        self.io.apb_write(A_M, M)
        self.io.apb_write(A_N, N)
        self.io.apb_write(A_K, K)
        self.io.apb_write(A_IFM, ifm_base)
        self.io.apb_write(A_W, w_base)
        self.io.apb_write(A_OFM, ofm_base)
        self.io.apb_write(A_BIAS, bias_base)
        self.io.apb_write(A_FLAGS, flags)
        self.io.apb_write(A_REQ_MULT, req_mult & 0xFFFF_FFFF)
        self.io.apb_write(A_REQ_SHIFT, req_shift)
        self.io.apb_write(A_REQ_MULT_BASE, req_mult_base)
        self.io.apb_write(A_REQ_SHIFT_BASE, req_shift_base)
        self.io.apb_write(A_CTRL, 1)

    def wait_done(self, max_polls: int = 10000):
        for _ in range(max_polls):
            status = self.io.apb_read(A_STATUS)
            busy = status & 0x1
            if not busy:
                return
        raise TimeoutError("tinyNPU did not complete within polling budget")

    def read_ofm(self, M: int, N: int, ofm_base: int) -> np.ndarray:
        n_tiles = N // self.cols
        out = np.zeros((M, N), dtype=np.int8)
        for n in range(n_tiles):
            for i in range(M):
                row = _unpack_i8_row(self.io.ofm_read(ofm_base + n * M + i), self.cols)
                out[i, n * self.cols : (n + 1) * self.cols] = np.array(row, dtype=np.int8)
        return out


if __name__ == "__main__":
    print("Use TinyNPUHost with your APB/SRAM backend; see source comments for flow.")
