"""Cocotb tests for rtl/systolic_array.sv (4x4 weight-stationary array).

Bus layout (LSB-first, row-major; matches RTL):
  w_in [(r*COLS + c)*8 +: 8] = W[r][c]   (signed INT8)
  a_in [r*8 +: 8]            = activation entering row r (signed INT8)
  psum_out[c*32 +: 32]       = partial sum exiting column c (signed INT32)

Driver schedule for a 4x4 GEMM C = A @ W (A is M x K=4, W is K=4 x N=4):
  - Stagger: row r is fed starting at cycle r.
  - At cycle t, row r's input = A[t-r, r] if 0 <= t-r < M else 0.
  - Output: psum_out[c] sampled after cycle (3 + c + i) -> C[i, c].
"""

from __future__ import annotations

import random
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ROWS = 4
COLS = 4
A_W = 8
P_W = 32

CLK_PERIOD_NS = 10
SETTLE_NS = 1


def s32(x: int) -> int:
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x & (1 << 31) else x


def pack_weights(w: np.ndarray) -> int:
    """w: shape (ROWS, COLS) int8 -> packed int (ROWS*COLS*8 bits)."""
    val = 0
    for r in range(ROWS):
        for c in range(COLS):
            byte = int(w[r, c]) & 0xFF
            val |= byte << ((r * COLS + c) * A_W)
    return val


def pack_activations(a_row: np.ndarray) -> int:
    """a_row: shape (ROWS,) int -> packed int (ROWS*8 bits)."""
    val = 0
    for r in range(ROWS):
        byte = int(a_row[r]) & 0xFF
        val |= byte << (r * A_W)
    return val


def unpack_psum(val: int, c: int) -> int:
    return s32((val >> (c * P_W)) & 0xFFFF_FFFF)


async def reset(dut, cycles: int = 2) -> None:
    dut.rst_n.value = 0
    dut.w_load.value = 0
    dut.w_in.value = 0
    dut.a_in.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def load_weights(dut, w: np.ndarray) -> None:
    dut.w_in.value = pack_weights(w)
    dut.w_load.value = 1
    await RisingEdge(dut.clk)
    dut.w_load.value = 0
    dut.w_in.value = 0
    await Timer(SETTLE_NS, units="ns")


async def run_gemm(dut, w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Run one staggered GEMM pass. Returns captured C of shape (M, COLS)."""
    M = a.shape[0]
    assert a.shape == (M, ROWS) and w.shape == (ROWS, COLS)

    await load_weights(dut, w)

    # Stagger schedule: at cycle t, row r feeds a[t-r, r] (if in range).
    n_cycles = M + ROWS + COLS - 1  # last C[M-1, COLS-1] at cycle 3 + (COLS-1) + (M-1)
    captured = np.zeros((n_cycles, COLS), dtype=np.int64)

    for t in range(n_cycles):
        a_vec = np.zeros(ROWS, dtype=np.int32)
        for r in range(ROWS):
            i = t - r
            if 0 <= i < M:
                a_vec[r] = int(a[i, r])
        dut.a_in.value = pack_activations(a_vec)
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")

        raw = int(dut.psum_out.value)
        for c in range(COLS):
            captured[t, c] = unpack_psum(raw, c)

    # Reorder captured into C: C[i, c] = captured[3 + c + i, c]
    result = np.zeros((M, COLS), dtype=np.int64)
    for c in range(COLS):
        for i in range(M):
            result[i, c] = captured[(ROWS - 1) + c + i, c]
    return result


@cocotb.test()
async def test_systolic_reset(dut):
    """All psum_out lanes are zero after reset."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)
    raw = int(dut.psum_out.value)
    for c in range(COLS):
        assert unpack_psum(raw, c) == 0, f"psum_out[{c}] = {unpack_psum(raw, c)}"


@cocotb.test()
async def test_systolic_identity_weights(dut):
    """W = I -> A @ I = A (zero-extended to INT32)."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    w = np.eye(ROWS, dtype=np.int8)
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [-1, -2, -3, -4],
                  [10, -20, 30, -40]], dtype=np.int8)
    expected = a.astype(np.int32) @ w.astype(np.int32)

    got = await run_gemm(dut, w, a)
    assert np.array_equal(got, expected), f"got=\n{got}\nexpected=\n{expected}"


@cocotb.test()
async def test_systolic_random_4x4(dut):
    """4x4 GEMM with random INT8 inputs, multiple seeds."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    for seed in [0xA, 0xBEEF, 0xC0FFEE, 1]:
        await reset(dut)
        rng = np.random.default_rng(seed)
        w = rng.integers(-128, 128, size=(ROWS, COLS), dtype=np.int8)
        a = rng.integers(-128, 128, size=(4, ROWS), dtype=np.int8)
        expected = a.astype(np.int32) @ w.astype(np.int32)
        got = await run_gemm(dut, w, a)
        assert np.array_equal(got, expected), (
            f"seed={seed}\nw=\n{w}\na=\n{a}\ngot=\n{got}\nexpected=\n{expected}"
        )


@cocotb.test()
async def test_systolic_extremes(dut):
    """INT8 extremes shouldn't overflow INT32 accumulator (4 terms max)."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    w = np.full((ROWS, COLS), -128, dtype=np.int8)
    a = np.full((4, ROWS), -128, dtype=np.int8)
    expected = a.astype(np.int32) @ w.astype(np.int32)
    got = await run_gemm(dut, w, a)
    assert np.array_equal(got, expected), f"got=\n{got}\nexpected=\n{expected}"


@cocotb.test()
async def test_systolic_back_to_back(dut):
    """Run two GEMMs back-to-back with different weights without resetting between."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x1234)

    for trial in range(2):
        w = rng.integers(-50, 50, size=(ROWS, COLS), dtype=np.int8)
        a = rng.integers(-50, 50, size=(4, ROWS), dtype=np.int8)
        expected = a.astype(np.int32) @ w.astype(np.int32)
        got = await run_gemm(dut, w, a)
        # After run_gemm, drain a few extra cycles before next w_load.
        dut.a_in.value = 0
        for _ in range(ROWS + COLS):
            await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert np.array_equal(got, expected), (
            f"trial={trial}\nw=\n{w}\na=\n{a}\ngot=\n{got}\nexpected=\n{expected}"
        )
