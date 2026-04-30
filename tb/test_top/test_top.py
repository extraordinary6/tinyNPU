"""End-to-end cocotb tests for tinyNPU_top.

Flow per test:
  1. backdoor-load A into IFM SRAM (each addr stores one ROWS x INT8 row)
  2. backdoor-load W tile into W SRAM at one address (ROWS*COLS x INT8 packed)
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
from golden_model import gemm_i8, bias_relu as model_bias_relu, requantize as model_req

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


async def read_ofm(dut, addr):
    dut.bd_ofm_re.value = 1
    dut.bd_ofm_addr.value = addr
    await RisingEdge(dut.pclk)
    dut.bd_ofm_re.value = 0
    await Timer(SETTLE_NS, units="ns")
    return int(dut.bd_ofm_rdata.value)


async def wait_done(dut, max_cycles=200):
    for _ in range(max_cycles):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.u_dut.busy.value) == 0:
            return
    raise TimeoutError("engine did not return to IDLE within max_cycles")


async def configure(dut, *, M, N, K, ifm_base, w_base, ofm_base,
                    flags=0, req_mult=1, req_shift=0):
    await apb_write(dut, A_M, M)
    await apb_write(dut, A_N, N)
    await apb_write(dut, A_K, K)
    await apb_write(dut, A_IFM, ifm_base)
    await apb_write(dut, A_W, w_base)
    await apb_write(dut, A_OFM, ofm_base)
    await apb_write(dut, A_FLAGS, flags)
    await apb_write(dut, A_REQ_MULT, req_mult & 0xFFFF_FFFF)
    await apb_write(dut, A_REQ_SHIFT, req_shift)


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
