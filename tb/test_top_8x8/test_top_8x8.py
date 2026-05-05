"""End-to-end cocotb tests for tinyNPU_top with ROWS=COLS=8 (8x8 PE array).

Mirrors tb/test_top/ but with the wider PE array. Confirms phase 12
parameterization works: stagger pipeline scales to ROWS=8, unskew scales
to COLS=8, valid_gen latency scales (target = ROWS + COLS - 1 = 15).
"""

from __future__ import annotations

import os
import sys
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "common"))
from golden_model import (
    gemm_i8,
    bias_relu as model_bias_relu,
    requantize as model_req,
    requantize_per_channel as model_req_pc,
)

CLK_NS = 10
SETTLE_NS = 1
ROWS = 8
COLS = 8

A_ID = 0x000
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


def s8(x):
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def pack_ifm_row(row):
    val = 0
    for r, v in enumerate(row):
        val |= (int(v) & 0xFF) << (r * 8)
    return val


def pack_w_tile(W):
    val = 0
    for r in range(ROWS):
        for c in range(COLS):
            val |= (int(W[r, c]) & 0xFF) << ((r * COLS + c) * 8)
    return val


def unpack_ofm_row(raw):
    return [s8((raw >> (c * 8)) & 0xFF) for c in range(COLS)]


def pack_bias_word(bias):
    val = 0
    for c, b in enumerate(bias):
        val |= (int(b) & 0xFFFF_FFFF) << (c * 32)
    return val


def pack_w_mults(mults):
    val = 0
    for c, m in enumerate(mults):
        val |= (int(m) & 0xFFFF_FFFF) << (c * 32)
    return val


def pack_w_shifts(shifts):
    val = 0
    for c, s in enumerate(shifts):
        val |= (int(s) & 0xFF) << (c * 32)
    return val


async def reset(dut, cycles=3):
    dut.presetn.value = 0
    dut.psel.value = 0
    dut.penable.value = 0
    dut.pwrite.value = 0
    dut.paddr.value = 0
    dut.pwdata.value = 0
    dut.bd_ifm_we.value = 0
    dut.bd_ifm_addr.value = 0
    dut.bd_ifm_wdata.value = 0
    dut.bd_w_we.value = 0
    dut.bd_w_addr.value = 0
    dut.bd_w_wdata.value = 0
    dut.bd_bias_we.value = 0
    dut.bd_bias_addr.value = 0
    dut.bd_bias_wdata.value = 0
    dut.bd_ofm_re.value = 0
    dut.bd_ofm_addr.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.pclk)
    dut.presetn.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")


async def apb_write(dut, addr, data):
    dut.psel.value = 1
    dut.penable.value = 0
    dut.pwrite.value = 1
    dut.paddr.value = addr
    dut.pwdata.value = data & 0xFFFF_FFFF
    await RisingEdge(dut.pclk)
    dut.penable.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")
    dut.psel.value = 0
    dut.penable.value = 0
    dut.pwrite.value = 0


async def apb_read(dut, addr):
    dut.psel.value = 1
    dut.penable.value = 0
    dut.pwrite.value = 0
    dut.paddr.value = addr
    await RisingEdge(dut.pclk)
    dut.penable.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")
    data = int(dut.prdata.value)
    dut.psel.value = 0
    dut.penable.value = 0
    return data


async def load_ifm(dut, A, base):
    """Single-K-tile IFM load (A is M x ROWS)."""
    for i, row in enumerate(A):
        dut.bd_ifm_we.value = 1
        dut.bd_ifm_addr.value = base + i
        dut.bd_ifm_wdata.value = pack_ifm_row(row)
        await RisingEdge(dut.pclk)
    dut.bd_ifm_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_ifm_ktile(dut, A, ifm_base, M):
    """Tile-major IFM: tile k slice A[:, k*ROWS:(k+1)*ROWS] at ifm_base + k*M + i."""
    K = A.shape[1]
    K_TILES = K // ROWS
    for k in range(K_TILES):
        slice_a = A[:, k * ROWS : (k + 1) * ROWS]
        for i in range(M):
            dut.bd_ifm_we.value = 1
            dut.bd_ifm_addr.value = ifm_base + k * M + i
            dut.bd_ifm_wdata.value = pack_ifm_row(slice_a[i])
            await RisingEdge(dut.pclk)
    dut.bd_ifm_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_w_full(dut, W, w_base):
    """W[K, N] split into (n_tile, k_tile) tiles at w_base + n*K_TILES + k."""
    K, N = W.shape
    K_TILES = K // ROWS
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        for k_tile in range(K_TILES):
            slab = W[k_tile * ROWS : (k_tile + 1) * ROWS,
                     n_tile * COLS : (n_tile + 1) * COLS]
            dut.bd_w_we.value = 1
            dut.bd_w_addr.value = w_base + n_tile * K_TILES + k_tile
            dut.bd_w_wdata.value = pack_w_tile(slab)
            await RisingEdge(dut.pclk)
    dut.bd_w_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_bias_full(dut, bias, bias_base):
    """bias is length N; one COLS x INT32 word per N tile at bias_base + n_tile."""
    N = len(bias)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        slab = bias[n_tile * COLS : (n_tile + 1) * COLS]
        dut.bd_bias_we.value = 1
        dut.bd_bias_addr.value = bias_base + n_tile
        dut.bd_bias_wdata.value = pack_bias_word(slab)
        await RisingEdge(dut.pclk)
    dut.bd_bias_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_w_word(dut, addr, packed):
    dut.bd_w_we.value = 1
    dut.bd_w_addr.value = addr
    dut.bd_w_wdata.value = packed
    await RisingEdge(dut.pclk)
    dut.bd_w_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_pch_full(dut, mults, shifts, mult_base, shift_base):
    """Per-channel mult/shift arrays of length N: one word per N tile."""
    N = len(mults)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        m_slab = mults[n_tile * COLS : (n_tile + 1) * COLS]
        s_slab = shifts[n_tile * COLS : (n_tile + 1) * COLS]
        await load_w_word(dut, addr=mult_base + n_tile, packed=pack_w_mults(m_slab))
        await load_w_word(dut, addr=shift_base + n_tile, packed=pack_w_shifts(s_slab))


async def read_ofm(dut, addr):
    dut.bd_ofm_re.value = 1
    dut.bd_ofm_addr.value = addr
    await RisingEdge(dut.pclk)
    dut.bd_ofm_re.value = 0
    await Timer(SETTLE_NS, units="ns")
    return int(dut.bd_ofm_rdata.value)


async def read_ofm_full(dut, ofm_base, M, N):
    out = np.zeros((M, N), dtype=np.int8)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        for i in range(M):
            raw = await read_ofm(dut, ofm_base + n_tile * M + i)
            row = unpack_ofm_row(raw)
            for c in range(COLS):
                out[i, n_tile * COLS + c] = row[c]
    return out


async def wait_done(dut, max_cycles=600):
    for _ in range(max_cycles):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.u_dut.busy.value) == 0:
            return
    raise TimeoutError("engine did not return to IDLE within max_cycles")


async def configure(dut, *, M, N, K, ifm_base, w_base, ofm_base,
                    flags=0, req_mult=1, req_shift=0,
                    req_mult_base=0, req_shift_base=0, bias_base=0):
    await apb_write(dut, A_M, M)
    await apb_write(dut, A_N, N)
    await apb_write(dut, A_K, K)
    await apb_write(dut, A_IFM, ifm_base)
    await apb_write(dut, A_W, w_base)
    await apb_write(dut, A_OFM, ofm_base)
    await apb_write(dut, A_BIAS, bias_base)
    await apb_write(dut, A_FLAGS, flags)
    await apb_write(dut, A_REQ_MULT, req_mult & 0xFFFF_FFFF)
    await apb_write(dut, A_REQ_SHIFT, req_shift)
    await apb_write(dut, A_REQ_MULT_BASE, req_mult_base)
    await apb_write(dut, A_REQ_SHIFT_BASE, req_shift_base)


async def kick(dut):
    await apb_write(dut, A_CTRL, 0x1)


@cocotb.test()
async def test_top_8x8_id(dut):
    """APB read of A_ID returns the magic constant."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    data = await apb_read(dut, A_ID)
    assert data == 0x4E50_5500


@cocotb.test()
async def test_top_8x8_gemm_only(dut):
    """8x8 GEMM (single tile each axis), raw INT32 low-byte output."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_01)
    M = 4
    N = 8
    K = 8
    A = rng.integers(-4, 4, size=(M, K), dtype=np.int8)
    W = rng.integers(-4, 4, size=(K, N), dtype=np.int8)

    # K=8=ROWS, single K tile -> use the simple flat IFM load.
    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x20,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    out_u8 = (acc & 0xFF).astype(np.uint8)
    expected = np.where(out_u8 & 0x80, out_u8.astype(np.int16) - 256, out_u8).astype(np.int8)
    got = await read_ofm_full(dut, 0x20, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_gemm_relu_req(dut):
    """8x8 GEMM + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_02)
    M = 4
    N = 8
    K = 8
    A = rng.integers(-16, 16, size=(M, K), dtype=np.int8)
    W = rng.integers(-16, 16, size=(K, N), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 20

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0b0110,
                    req_mult=req_mult, req_shift=req_shift)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    acc = model_bias_relu(acc, None, bias_en=False, relu_en=True)
    expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    got = await read_ofm_full(dut, 0x40, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_ktile_k16(dut):
    """K=16 (2 K tiles) on 8x8 array."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_03)
    M = 4
    N = 8
    K = 16
    A = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    W = rng.integers(-8, 8, size=(K, N), dtype=np.int8)

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    out_u8 = (acc & 0xFF).astype(np.uint8)
    expected = np.where(out_u8 & 0x80, out_u8.astype(np.int16) - 256, out_u8).astype(np.int8)
    got = await read_ofm_full(dut, 0x40, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_ntile_n16(dut):
    """N=16 (2 N tiles) on 8x8 array, raw output."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_04)
    M = 4
    N = 16
    K = 8
    A = rng.integers(-4, 4, size=(M, K), dtype=np.int8)
    W = rng.integers(-4, 4, size=(K, N), dtype=np.int8)

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    out_u8 = (acc & 0xFF).astype(np.uint8)
    expected = np.where(out_u8 & 0x80, out_u8.astype(np.int16) - 256, out_u8).astype(np.int8)
    got = await read_ofm_full(dut, 0x40, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_full(dut):
    """N=16, K=16, M=8 — nested N+K loops on 8x8 array, with bias + ReLU + req."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_FF)
    M = 8
    N = 16
    K = 16
    A = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    W = rng.integers(-8, 8, size=(K, N), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-200, 200, size=N, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 22
    bias_addr = 0x10

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_full(dut, W, w_base=0)
    await load_bias_full(dut, bias, bias_base=bias_addr)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x60,
                    flags=0b0111,
                    req_mult=req_mult, req_shift=req_shift,
                    bias_base=bias_addr)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    bias_vec = np.array(bias, dtype=np.int32)
    bias_bcast = np.broadcast_to(bias_vec, acc.shape)
    acc = model_bias_relu(acc, bias_bcast, bias_en=True, relu_en=True)
    expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    got = await read_ofm_full(dut, 0x60, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_per_channel(dut):
    """8x8 GEMM + per-channel requantize. mult/shift per-channel from W SRAM."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x88_C8)
    M = 4
    N = 8
    K = 8
    A = rng.integers(-32, 32, size=(M, K), dtype=np.int8)
    W = rng.integers(-32, 32, size=(K, N), dtype=np.int8)
    mults = [(1 << 18), (1 << 19), -(1 << 17), (1 << 16),
             (1 << 20), -(1 << 18), (1 << 17), (1 << 21)]
    shifts = [12, 14, 11, 10, 16, 13, 12, 17]

    N_TILES = N // COLS
    K_TILES = K // ROWS
    w_base = 0
    mult_base = w_base + N_TILES * K_TILES
    shift_base = mult_base + N_TILES

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=w_base)
    await load_pch_full(dut, mults, shifts, mult_base=mult_base, shift_base=shift_base)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=w_base, ofm_base=0x40,
                    flags=0b1100,
                    req_mult=0, req_shift=0,
                    req_mult_base=mult_base, req_shift_base=shift_base)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    expected = model_req_pc(acc, mults, shifts)
    got = await read_ofm_full(dut, 0x40, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_8x8_cycle_count(dut):
    """Measure the start-to-busy=0 cycle count for a single-tile 8x8 GEMM.

    Used as the data point for docs/perf.md. Compare against the 4x4
    reference (N=K=4, M=4) which takes 26 cycles end-to-end.

    Theoretical lower bound for ROWS=COLS=L, single tile, M rows:
      M (compute beats)
      + (ROWS-1)         (stagger fill, top row first vs bottom row last)
      + (COLS-1)         (rightmost column needs ROWS-1 horizontal hops)
      + 2                (FSM startup + SRAM read)
      + few              (writeback drain)
    """
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    M = 4
    N = 8
    K = 8
    A = np.zeros((M, K), dtype=np.int8)
    W = np.zeros((K, N), dtype=np.int8)
    A[:] = 1
    W[:] = 1

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x80,
                    flags=0, req_mult=1, req_shift=0)

    # Kick and count cycles until busy goes back to 0.
    await apb_write(dut, A_CTRL, 0x1)
    cycles = 0
    seen_busy = False
    for _ in range(400):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        cycles += 1
        if int(dut.u_dut.busy.value):
            seen_busy = True
        elif seen_busy:
            break
    assert seen_busy, "engine never went busy"
    cocotb.log.info(f"8x8 GEMM (M={M}, N={N}, K={K}) cycles to busy=0: {cycles}")

    # Sanity check the result while we're here.
    expected_value = K  # all ones, K elements summed
    for i in range(M):
        raw = await read_ofm(dut, 0x80 + i)
        row = unpack_ofm_row(raw)
        for c in range(COLS):
            assert row[c] == (expected_value & 0xFF) - (0x100 if (expected_value & 0x80) else 0), \
                f"row {i} col {c}: got {row[c]}"


# ---------------- phase 14 cycle-count sweeps ----------------
# Same shape as the 4x4 sweeps in tb/test_top/test_top.py: vary M to
# show fill/drain pipeline tax amortising over output rows. Single-tile
# (N=K=8) and multi-tile (N=K=16) variants. Numbers feed docs/perf.md.

async def _measure_cycles(dut, max_cycles=4000):
    await apb_write(dut, A_CTRL, 0x1)
    cycles = 0
    seen_busy = False
    for _ in range(max_cycles):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        cycles += 1
        if int(dut.u_dut.busy.value):
            seen_busy = True
        elif seen_busy:
            return cycles
    raise TimeoutError("engine never returned to IDLE")


async def _run_8x8_singletile(dut, M):
    """Single (n_tile=1, k_tile=1) measurement at the requested M on 8x8."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    N = COLS
    K = ROWS
    A = np.ones((M, K), dtype=np.int8)
    W = np.ones((K, N), dtype=np.int8)
    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x80,
                    flags=0, req_mult=1, req_shift=0)
    cycles = await _measure_cycles(dut)
    macs = M * N * K
    cocotb.log.info(
        f"PERF 8x8 single-tile M={M} N={N} K={K}: "
        f"cycles={cycles} macs={macs} mac_per_cycle={macs/cycles:.3f} "
        f"util={100*macs/(cycles*ROWS*COLS):.1f}%"
    )


async def _run_8x8_n16k16(dut, M):
    """N=16, K=16 (2 N tiles x 2 K tiles) at the requested M on 8x8."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    N = 16
    K = 16
    A = np.ones((M, K), dtype=np.int8)
    W = np.ones((K, N), dtype=np.int8)
    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x80,
                    flags=0, req_mult=1, req_shift=0)
    cycles = await _measure_cycles(dut)
    macs = M * N * K
    cocotb.log.info(
        f"PERF 8x8 ntile=2 ktile=2 M={M} N={N} K={K}: "
        f"cycles={cycles} macs={macs} mac_per_cycle={macs/cycles:.3f} "
        f"util={100*macs/(cycles*ROWS*COLS):.1f}%"
    )


@cocotb.test()
async def test_top_8x8_perf_singletile_m8(dut):
    await _run_8x8_singletile(dut, M=8)


@cocotb.test()
async def test_top_8x8_perf_singletile_m16(dut):
    await _run_8x8_singletile(dut, M=16)


@cocotb.test()
async def test_top_8x8_perf_n16k16_m8(dut):
    await _run_8x8_n16k16(dut, M=8)


@cocotb.test()
async def test_top_8x8_perf_n16k16_m16(dut):
    await _run_8x8_n16k16(dut, M=16)
