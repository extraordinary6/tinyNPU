"""End-to-end cocotb tests for tinyNPU_top.

Flow per test:
  1. backdoor-load A into IFM SRAM (each addr stores one ROWS x INT8 row)
  2. backdoor-load W tile(s) into W SRAM (per-(n_tile, k_tile) ROWS*COLS x INT8 packed)
  3. APB writes M, N, K, base addresses, FLAGS, REQ_MULT, REQ_SHIFT
  4. APB writes CTRL[0]=1 to start
  5. poll busy until 0 (engine returned to IDLE)
  6. backdoor-read OFM SRAM, compare to numpy golden model
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
ROWS = 4
COLS = 4

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
    for i, row in enumerate(A):
        dut.bd_ifm_we.value = 1
        dut.bd_ifm_addr.value = base + i
        dut.bd_ifm_wdata.value = pack_ifm_row(row)
        await RisingEdge(dut.pclk)
    dut.bd_ifm_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def load_w(dut, W, addr):
    dut.bd_w_we.value = 1
    dut.bd_w_addr.value = addr
    dut.bd_w_wdata.value = pack_w_tile(W)
    await RisingEdge(dut.pclk)
    dut.bd_w_we.value = 0
    await Timer(SETTLE_NS, units="ns")


def pack_bias_word(bias):
    """Pack LANES INT32 bias values into one COLS*P_W-bit SRAM word."""
    val = 0
    for c, b in enumerate(bias):
        val |= (int(b) & 0xFFFF_FFFF) << (c * 32)
    return val


async def load_bias(dut, bias, addr):
    dut.bd_bias_we.value = 1
    dut.bd_bias_addr.value = addr
    dut.bd_bias_wdata.value = pack_bias_word(bias)
    await RisingEdge(dut.pclk)
    dut.bd_bias_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def read_ofm(dut, addr):
    dut.bd_ofm_re.value = 1
    dut.bd_ofm_addr.value = addr
    await RisingEdge(dut.pclk)
    dut.bd_ofm_re.value = 0
    await Timer(SETTLE_NS, units="ns")
    return int(dut.bd_ofm_rdata.value)


async def wait_done(dut, max_cycles=400):
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


def expected_output(A, W, *, relu_en, req_en, req_mult, req_shift):
    acc = gemm_i8(A.astype(np.int8), W.astype(np.int8))
    acc = model_bias_relu(acc, None, bias_en=False, relu_en=relu_en)
    if req_en:
        acc = model_req(acc, int(req_mult), int(req_shift))
        return acc.astype(np.int8)
    out = (acc & 0xFF).astype(np.uint8)
    return np.where(out & 0x80, out.astype(np.int16) - 256, out).astype(np.int8)


@cocotb.test()
async def test_top_id(dut):
    """APB read of A_ID returns the magic constant."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    data = await apb_read(dut, A_ID)
    assert data == 0x4E50_5500


@cocotb.test()
async def test_top_gemm_only(dut):
    """4x4 GEMM, no relu, no requantize. Output is INT8 = low byte of INT32."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xA1)
    M = 4
    A = rng.integers(-8, 8, size=(M, ROWS), dtype=np.int8)
    W = rng.integers(-8, 8, size=(ROWS, COLS), dtype=np.int8)

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=0)
    await configure(dut, M=M, N=COLS, K=ROWS,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    await wait_done(dut)

    expected = expected_output(A, W, relu_en=False, req_en=False, req_mult=1, req_shift=0)
    for i in range(M):
        raw = await read_ofm(dut, 0x40 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_gemm_relu_requantize(dut):
    """GEMM + ReLU + requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xBEEF)
    M = 4
    A = rng.integers(-128, 128, size=(M, ROWS), dtype=np.int8)
    W = rng.integers(-32, 32, size=(ROWS, COLS), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 24

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=0)
    await configure(dut, M=M, N=COLS, K=ROWS,
                    ifm_base=0, w_base=0, ofm_base=0x10,
                    flags=0b110, req_mult=req_mult, req_shift=req_shift)
    await kick(dut)
    await wait_done(dut)

    expected = expected_output(A, W, relu_en=True, req_en=True,
                               req_mult=req_mult, req_shift=req_shift)
    for i in range(M):
        raw = await read_ofm(dut, 0x10 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_back_to_back(dut):
    """Two GEMM kicks back-to-back at distinct addresses."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0DE)
    for trial in range(2):
        M = 4
        A = rng.integers(-16, 16, size=(M, ROWS), dtype=np.int8)
        W = rng.integers(-16, 16, size=(ROWS, COLS), dtype=np.int8)
        ifm_base = trial * 0x10
        w_addr = trial * 0x4
        ofm_base = 0x80 + trial * 0x10

        await load_ifm(dut, A, base=ifm_base)
        await load_w(dut, W, addr=w_addr)
        await configure(dut, M=M, N=COLS, K=ROWS,
                        ifm_base=ifm_base, w_base=w_addr, ofm_base=ofm_base,
                        flags=0b100, req_mult=1, req_shift=0)
        await kick(dut)
        await wait_done(dut)

        expected = expected_output(A, W, relu_en=False, req_en=True,
                                   req_mult=1, req_shift=0)
        for i in range(M):
            raw = await read_ofm(dut, ofm_base + i)
            got = unpack_ofm_row(raw)
            exp = [int(v) for v in expected[i]]
            assert got == exp, f"trial={trial} row={i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_err_on_zero_dim(dut):
    """M=0 + start should pulse ERR briefly without writing OFM."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    await configure(dut, M=0, N=COLS, K=ROWS,
                    ifm_base=0, w_base=0, ofm_base=0,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    saw_err = 0
    for _ in range(8):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.u_dut.err.value):
            saw_err += 1
    assert saw_err >= 1, "ERR was never asserted"
    assert int(dut.u_dut.busy.value) == 0


def pack_w_mults(mults):
    """Place LANES INT32 mults into one packed ROWS*COLS*INT8 word in W SRAM."""
    val = 0
    for c, m in enumerate(mults):
        val |= (int(m) & 0xFFFF_FFFF) << (c * 32)
    return val


def pack_w_shifts(shifts):
    """LANES shifts in low byte of each 32-bit lane slot."""
    val = 0
    for c, s in enumerate(shifts):
        val |= (int(s) & 0xFF) << (c * 32)
    return val


async def load_w_word(dut, addr, packed):
    dut.bd_w_we.value = 1
    dut.bd_w_addr.value = addr
    dut.bd_w_wdata.value = packed
    await RisingEdge(dut.pclk)
    dut.bd_w_we.value = 0
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_top_per_channel_requantize(dut):
    """Per-channel requantize: FLAGS[3] enables LOAD_REQ; W SRAM holds mult/shift."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x8C8)
    M = 4
    A = rng.integers(-32, 32, size=(M, ROWS), dtype=np.int8)
    W = rng.integers(-32, 32, size=(ROWS, COLS), dtype=np.int8)

    mults = [1 << 18, 1 << 20, -(1 << 19), 1 << 17]
    shifts = [12, 14, 11, 10]

    w_addr = 0
    mult_addr = 0x20
    shift_addr = 0x21

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=w_addr)
    await load_w_word(dut, addr=mult_addr, packed=pack_w_mults(mults))
    await load_w_word(dut, addr=shift_addr, packed=pack_w_shifts(shifts))

    # FLAGS[3]=PCH_REQ_EN, FLAGS[2]=REQ_EN, FLAGS[1]=RELU_EN.
    await configure(dut, M=M, N=COLS, K=ROWS,
                    ifm_base=0, w_base=w_addr, ofm_base=0x60,
                    flags=0b1100,
                    req_mult=0, req_shift=0,
                    req_mult_base=mult_addr, req_shift_base=shift_addr)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A.astype(np.int8), W.astype(np.int8))
    expected = model_req_pc(acc, mults, shifts)
    for i in range(M):
        raw = await read_ofm(dut, 0x60 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_bias_relu_requantize(dut):
    """GEMM + bias + ReLU + requantize, with bias loaded from BIAS SRAM."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xB1A5)
    M = 4
    A = rng.integers(-32, 32, size=(M, ROWS), dtype=np.int8)
    W = rng.integers(-32, 32, size=(ROWS, COLS), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-500, 500, size=COLS, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 18
    bias_addr = 0x05

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=0)
    await load_bias(dut, bias, addr=bias_addr)
    # FLAGS[2]=REQ_EN, [1]=RELU_EN, [0]=BIAS_EN.
    await configure(dut, M=M, N=COLS, K=ROWS,
                    ifm_base=0, w_base=0, ofm_base=0x70,
                    flags=0b0111,
                    req_mult=req_mult, req_shift=req_shift,
                    bias_base=bias_addr)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A.astype(np.int8), W.astype(np.int8))
    bias_vec = np.array(bias, dtype=np.int32)
    bias_bcast = np.broadcast_to(bias_vec, acc.shape)
    acc = model_bias_relu(acc, bias_bcast, bias_en=True, relu_en=True)
    expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    for i in range(M):
        raw = await read_ofm(dut, 0x70 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_bias_only(dut):
    """GEMM + bias (no ReLU, no requantize) — output is low byte of (acc + bias)."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xB1A5_F00D)
    M = 4
    A = rng.integers(-4, 4, size=(M, ROWS), dtype=np.int8)
    W = rng.integers(-4, 4, size=(ROWS, COLS), dtype=np.int8)
    bias = [10, -20, 30, -40]
    bias_addr = 0x07

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=0)
    await load_bias(dut, bias, addr=bias_addr)
    await configure(dut, M=M, N=COLS, K=ROWS,
                    ifm_base=0, w_base=0, ofm_base=0x90,
                    flags=0b0001,
                    req_mult=1, req_shift=0,
                    bias_base=bias_addr)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A.astype(np.int8), W.astype(np.int8))
    bias_vec = np.array(bias, dtype=np.int32)
    sum_int32 = acc + np.broadcast_to(bias_vec, acc.shape)
    out = (sum_int32 & 0xFF).astype(np.uint8)
    expected = np.where(out & 0x80, out.astype(np.int16) - 256, out).astype(np.int8)
    for i in range(M):
        raw = await read_ofm(dut, 0x90 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"row {i}: got={got} expected={exp}"


async def load_ifm_ktile(dut, A, ifm_base, M):
    """Load A[M, K] into IFM SRAM with tile-major layout: tile k slice
    A[:, k*ROWS:(k+1)*ROWS] occupies addresses ifm_base + k*M ... + k*M + M-1."""
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


async def load_w_ktile(dut, W, w_base):
    """Load W[K, COLS] into W SRAM, tile k = W[k*ROWS:(k+1)*ROWS, :] at w_base + k."""
    K = W.shape[0]
    K_TILES = K // ROWS
    for k in range(K_TILES):
        slice_w = W[k * ROWS : (k + 1) * ROWS, :]
        dut.bd_w_we.value = 1
        dut.bd_w_addr.value = w_base + k
        dut.bd_w_wdata.value = pack_w_tile(slice_w)
        await RisingEdge(dut.pclk)
    dut.bd_w_we.value = 0
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_top_ktile_k8(dut):
    """K=8 GEMM via two 4-wide tiles. raw INT32 low-byte output."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x10E_8)
    M = 4
    K = 8
    A = rng.integers(-4, 4, size=(M, K), dtype=np.int8)
    W = rng.integers(-4, 4, size=(K, COLS), dtype=np.int8)

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_ktile(dut, W, w_base=0)
    await configure(dut, M=M, N=COLS, K=K,
                    ifm_base=0, w_base=0, ofm_base=0xA0,
                    flags=0, req_mult=1, req_shift=0)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    out_u8 = (acc & 0xFF).astype(np.uint8)
    expected = np.where(out_u8 & 0x80, out_u8.astype(np.int16) - 256, out_u8).astype(np.int8)
    for i in range(M):
        raw = await read_ofm(dut, 0xA0 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"K=8 row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_ktile_k12_relu_req(dut):
    """K=12 (3 tiles) GEMM + ReLU + requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x10E_C)
    M = 4
    K = 12
    A = rng.integers(-16, 16, size=(M, K), dtype=np.int8)
    W = rng.integers(-16, 16, size=(K, COLS), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 22

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_ktile(dut, W, w_base=0)
    await configure(dut, M=M, N=COLS, K=K,
                    ifm_base=0, w_base=0, ofm_base=0xC0,
                    flags=0b0110,
                    req_mult=req_mult, req_shift=req_shift)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    acc = model_bias_relu(acc, None, bias_en=False, relu_en=True)
    expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    for i in range(M):
        raw = await read_ofm(dut, 0xC0 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"K=12 row {i}: got={got} expected={exp}"


@cocotb.test()
async def test_top_ktile_k8_bias_relu_req(dut):
    """K=8 GEMM + bias + ReLU + requantize — bias loaded once on first tile."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x10E_88)
    M = 4
    K = 8
    A = rng.integers(-16, 16, size=(M, K), dtype=np.int8)
    W = rng.integers(-16, 16, size=(K, COLS), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-200, 200, size=COLS, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 20
    bias_addr = 0x09

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_ktile(dut, W, w_base=0)
    await load_bias(dut, bias, addr=bias_addr)
    await configure(dut, M=M, N=COLS, K=K,
                    ifm_base=0, w_base=0, ofm_base=0xE0,
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
    for i in range(M):
        raw = await read_ofm(dut, 0xE0 + i)
        got = unpack_ofm_row(raw)
        exp = [int(v) for v in expected[i]]
        assert got == exp, f"K=8+bias row {i}: got={got} expected={exp}"


# ----------------------------------------------------------------------
# Phase 11: N-tile sweep helpers
# ----------------------------------------------------------------------

async def load_w_full(dut, W, w_base):
    """Load W[K, N] into W SRAM. Layout: tile (n_tile, k_tile) at
    w_base + n_tile*K_TILES + k_tile, slab W[k*ROWS:(k+1)*ROWS, n*COLS:(n+1)*COLS]."""
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
    """Load length-N bias array as N_TILES words at bias_base + n_tile."""
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


async def load_pch_full(dut, mults, shifts, mult_base, shift_base):
    """Per-channel mult and shift arrays of length N: one word per N tile."""
    N = len(mults)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        m_slab = mults[n_tile * COLS : (n_tile + 1) * COLS]
        s_slab = shifts[n_tile * COLS : (n_tile + 1) * COLS]
        await load_w_word(dut, addr=mult_base + n_tile, packed=pack_w_mults(m_slab))
        await load_w_word(dut, addr=shift_base + n_tile, packed=pack_w_shifts(s_slab))


async def read_ofm_full(dut, ofm_base, M, N):
    """Read M x N output back. Tile n's M rows live at ofm_base + n*M + i."""
    out = np.zeros((M, N), dtype=np.int8)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        for i in range(M):
            raw = await read_ofm(dut, ofm_base + n_tile * M + i)
            row = unpack_ofm_row(raw)
            for c in range(COLS):
                out[i, n_tile * COLS + c] = row[c]
    return out


@cocotb.test()
async def test_top_ntile_n8(dut):
    """N=8 GEMM (two N tiles), K=4. Raw INT32 low-byte output."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_8)
    M = 4
    N = 8
    K = 4
    A = rng.integers(-8, 8, size=(M, K), dtype=np.int8)
    W = rng.integers(-8, 8, size=(K, N), dtype=np.int8)

    # K=4 -> single K tile, so IFM layout is the simple flat one (M rows).
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
async def test_top_ntile_n8_relu_req(dut):
    """N=8 GEMM + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_8C)
    M = 4
    N = 8
    K = 4
    A = rng.integers(-32, 32, size=(M, K), dtype=np.int8)
    W = rng.integers(-32, 32, size=(K, N), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 18

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x60,
                    flags=0b0110,
                    req_mult=req_mult, req_shift=req_shift)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    acc = model_bias_relu(acc, None, bias_en=False, relu_en=True)
    expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    got = await read_ofm_full(dut, 0x60, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_ntile_n8_bias_relu_req(dut):
    """N=8 GEMM + per-N-tile bias + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_88)
    M = 4
    N = 8
    K = 4
    A = rng.integers(-16, 16, size=(M, K), dtype=np.int8)
    W = rng.integers(-16, 16, size=(K, N), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-300, 300, size=N, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 18
    bias_addr = 0x10

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=0)
    await load_bias_full(dut, bias, bias_base=bias_addr)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x80,
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
    got = await read_ofm_full(dut, 0x80, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_ntile_n8_per_channel(dut):
    """N=8 GEMM + per-channel requantize (mult/shift loaded per N tile from W SRAM)."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_C0)
    M = 4
    N = 8
    K = 4
    A = rng.integers(-32, 32, size=(M, K), dtype=np.int8)
    W = rng.integers(-32, 32, size=(K, N), dtype=np.int8)
    mults = [(1 << 18), (1 << 19), -(1 << 18), (1 << 17),
             (1 << 20), -(1 << 19), (1 << 18), (1 << 16)]
    shifts = [12, 14, 11, 10, 15, 13, 12, 9]

    # W SRAM layout: weight tiles at w_base..w_base + N_TILES*K_TILES - 1,
    # then mult words at mult_base..mult_base + N_TILES - 1, then shift
    # words at shift_base..shift_base + N_TILES - 1.
    N_TILES = N // COLS
    K_TILES = K // ROWS
    w_base = 0
    mult_base = w_base + N_TILES * K_TILES
    shift_base = mult_base + N_TILES

    await load_ifm(dut, A, base=0)
    await load_w_full(dut, W, w_base=w_base)
    await load_pch_full(dut, mults, shifts, mult_base=mult_base, shift_base=shift_base)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=w_base, ofm_base=0xA0,
                    flags=0b1100,
                    req_mult=0, req_shift=0,
                    req_mult_base=mult_base, req_shift_base=shift_base)
    await kick(dut)
    await wait_done(dut)

    acc = gemm_i8(A, W)
    expected = model_req_pc(acc, mults, shifts)
    got = await read_ofm_full(dut, 0xA0, M, N)
    np.testing.assert_array_equal(got, expected)


@cocotb.test()
async def test_top_ntile_n16(dut):
    """N=16 GEMM (four N tiles), K=4. Larger N sweep, no extras."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_F)
    M = 2
    N = 16
    K = 4
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
async def test_top_ntile_n8_ktile_k8_full(dut):
    """N=8, K=8, M=4 — nested N+K loops, with bias + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0x11E_AB)
    M = 4
    N = 8
    K = 8
    A = rng.integers(-16, 16, size=(M, K), dtype=np.int8)
    W = rng.integers(-16, 16, size=(K, N), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-300, 300, size=N, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 22
    bias_addr = 0x20

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
async def test_top_4x4_cycle_count(dut):
    """Measure start-to-busy=0 cycle count for a single-tile 4x4 GEMM (M=4).

    Used as the 4x4 data point for docs/perf.md. Pairs with the 8x8
    measurement in tb/test_top_8x8/test_top_8x8.py.
    """
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    M = 4
    N = COLS
    K = ROWS
    A = np.ones((M, K), dtype=np.int8)
    W = np.ones((K, N), dtype=np.int8)

    await load_ifm(dut, A, base=0)
    await load_w(dut, W, addr=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0, req_mult=1, req_shift=0)

    await apb_write(dut, A_CTRL, 0x1)
    cycles = 0
    seen_busy = False
    for _ in range(200):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        cycles += 1
        if int(dut.u_dut.busy.value):
            seen_busy = True
        elif seen_busy:
            break
    assert seen_busy
    cocotb.log.info(f"4x4 GEMM (M={M}, N={N}, K={K}) cycles to busy=0: {cycles}")


@cocotb.test()
async def test_top_4x4_cycle_count_n8k8(dut):
    """Measure cycle count for the M=4, N=8, K=8 problem on the 4x4 array.

    Same problem as test_top_8x8_cycle_count but lowered onto a 4x4
    array, so the engine has to walk 2 N tiles x 2 K tiles = 4 tiles.
    Used for the perf comparison in docs/perf.md.
    """
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    M = 4
    N = 8
    K = 8
    A = np.ones((M, K), dtype=np.int8)
    W = np.ones((K, N), dtype=np.int8)

    await load_ifm_ktile(dut, A, ifm_base=0, M=M)
    await load_w_full(dut, W, w_base=0)
    await configure(dut, M=M, N=N, K=K,
                    ifm_base=0, w_base=0, ofm_base=0x40,
                    flags=0, req_mult=1, req_shift=0)

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
    assert seen_busy
    cocotb.log.info(f"4x4 GEMM (M={M}, N={N}, K={K}) cycles to busy=0: {cycles}")
