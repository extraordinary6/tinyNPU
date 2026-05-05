"""End-to-end Conv2D cocotb tests for tinyNPU_top (phase 13).

Conv2D is run entirely in software on top of the existing INT8 GEMM
engine: the testbench im2col-expands IFM and reshapes the kernel into
A[M,K] @ B[K,N], pushes both through the backdoor SRAM ports just like
test_top, then compares the OFM against the pure-numpy
conv2d_reference (independent nested-loop implementation, not im2col).

GEMM mapping:
    M = H' * W'        (output rows in row-major (i, j) order)
    K = kh * kw * Cin  (must be a multiple of COLS = 4)
    N = Cout           (must be a multiple of COLS = 4)
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
    bias_relu as model_bias_relu,
    requantize as model_req,
    requantize_per_channel as model_req_pc,
)
from im2col import (
    im2col,
    kernel_to_b,
    output_shape,
    conv2d_reference,
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


def unpack_ofm_row(raw):
    return [s8((raw >> (c * 8)) & 0xFF) for c in range(COLS)]


# ---------------- APB / reset / wait helpers ----------------


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


async def wait_done(dut, max_cycles=4000):
    for _ in range(max_cycles):
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.u_dut.busy.value) == 0:
            return
    raise TimeoutError("engine did not return to IDLE within max_cycles")


# ---------------- backdoor SRAM helpers (M-tile/K-tile/N-tile aware) ----------------


async def load_ifm_ktile(dut, A, ifm_base, M):
    """A[M, K] tile-major: tile k slice A[:, k*ROWS:(k+1)*ROWS] -> ifm_base + k*M + i."""
    K = A.shape[1]
    assert K % ROWS == 0, f"K={K} must be a multiple of ROWS={ROWS}"
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
    """W[K, N] tile-major over (n_tile, k_tile): slab at w_base + n*K_TILES + k."""
    K, N = W.shape
    assert K % ROWS == 0 and N % COLS == 0
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
    """Length-N bias array: one COLS x INT32 word per N tile."""
    N = len(bias)
    assert N % COLS == 0
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
    """Per-channel mult/shift arrays (length N): one word per N tile in W SRAM."""
    N = len(mults)
    assert N % COLS == 0
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
    """Read M x N output back. Tile n's M rows live at ofm_base + n*M + i."""
    assert N % COLS == 0
    out = np.zeros((M, N), dtype=np.int8)
    N_TILES = N // COLS
    for n_tile in range(N_TILES):
        for i in range(M):
            raw = await read_ofm(dut, ofm_base + n_tile * M + i)
            row = unpack_ofm_row(raw)
            for c in range(COLS):
                out[i, n_tile * COLS + c] = row[c]
    return out


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


def low_byte_signed(acc_i32: np.ndarray) -> np.ndarray:
    """RTL raw-INT32 bypass output is the low byte of each accumulator lane."""
    out_u8 = (acc_i32 & 0xFF).astype(np.uint8)
    return np.where(out_u8 & 0x80,
                    out_u8.astype(np.int16) - 256,
                    out_u8).astype(np.int8)


# ---------------- shared conv driver ----------------


async def run_conv(dut, ifm, kernel, *,
                   stride=1, padding=0,
                   relu_en=False, req_en=False,
                   req_mult=1, req_shift=0,
                   bias=None,
                   per_channel=False, mults=None, shifts=None,
                   ifm_base=0, w_base=0, ofm_base=0x80, bias_base=0x10):
    """im2col -> load -> kick -> read -> compare against conv2d_reference.

    Returns the accumulator-domain INT32 reference and the readback INT8
    output for callers that want extra checks.
    """
    H, W, Cin = ifm.shape
    kh, kw, _, Cout = kernel.shape
    Hp, Wp = output_shape(H, W, kh, kw, stride, padding)
    M = Hp * Wp
    K = kh * kw * Cin
    N = Cout
    assert K % ROWS == 0, f"K={K} must be a multiple of {ROWS} (kh*kw*Cin)"
    assert N % COLS == 0, f"Cout={N} must be a multiple of {COLS}"

    A = im2col(ifm, kh, kw, stride, padding)
    B = kernel_to_b(kernel)

    flags = (0x1 if bias is not None else 0) \
          | (0x2 if relu_en else 0) \
          | (0x4 if req_en else 0) \
          | (0x8 if per_channel else 0)

    # W SRAM layout: weight tiles first, then (optional) mult/shift words.
    K_TILES = K // ROWS
    N_TILES = N // COLS
    mult_base = w_base + N_TILES * K_TILES
    shift_base = mult_base + N_TILES

    await load_ifm_ktile(dut, A, ifm_base=ifm_base, M=M)
    await load_w_full(dut, B, w_base=w_base)
    if bias is not None:
        await load_bias_full(dut, bias, bias_base=bias_base)
    if per_channel:
        assert mults is not None and shifts is not None
        await load_pch_full(dut, mults, shifts,
                            mult_base=mult_base, shift_base=shift_base)

    await configure(dut, M=M, N=N, K=K,
                    ifm_base=ifm_base, w_base=w_base, ofm_base=ofm_base,
                    flags=flags,
                    req_mult=req_mult, req_shift=req_shift,
                    req_mult_base=mult_base if per_channel else 0,
                    req_shift_base=shift_base if per_channel else 0,
                    bias_base=bias_base)
    await kick(dut)
    await wait_done(dut)

    # Reference path: independent nested-loop conv -> bias -> ReLU -> requant.
    ref = conv2d_reference(ifm, kernel, stride, padding)  # [Hp, Wp, Cout]
    acc = ref.reshape(M, N)
    if bias is not None:
        bias_vec = np.array(bias, dtype=np.int32)
        bias_bcast = np.broadcast_to(bias_vec, acc.shape)
        acc = model_bias_relu(acc, bias_bcast, bias_en=True, relu_en=relu_en)
    elif relu_en:
        acc = model_bias_relu(acc, None, bias_en=False, relu_en=True)
    if req_en:
        if per_channel:
            expected = model_req_pc(acc, mults, shifts)
        else:
            expected = model_req(acc, int(req_mult), int(req_shift)).astype(np.int8)
    else:
        expected = low_byte_signed(acc)

    got = await read_ofm_full(dut, ofm_base, M, N)
    np.testing.assert_array_equal(got, expected)
    return acc, got


# ---------------- tests ----------------


@cocotb.test()
async def test_conv_id(dut):
    """Sanity: APB ID register reads the magic constant."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    data = await apb_read(dut, A_ID)
    assert data == 0x4E50_5500


@cocotb.test()
async def test_conv_4x4_k2x2(dut):
    """Small conv: H=W=4, kh=kw=2, Cin=Cout=4, stride=1, no padding.
    M=9, K=16, N=4. Raw INT32 low-byte output (no requantize)."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_4422)
    ifm = rng.integers(-8, 8, size=(4, 4, 4), dtype=np.int8)
    ker = rng.integers(-8, 8, size=(2, 2, 4, 4), dtype=np.int8)

    await run_conv(dut, ifm, ker, stride=1, padding=0,
                   ofm_base=0x40)


@cocotb.test()
async def test_conv_4x4_k2x2_stride2(dut):
    """Strided conv: stride=2 -> H'=W'=2, M=4, K=16, N=4."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_5202)
    ifm = rng.integers(-16, 16, size=(4, 4, 4), dtype=np.int8)
    ker = rng.integers(-16, 16, size=(2, 2, 4, 4), dtype=np.int8)

    await run_conv(dut, ifm, ker, stride=2, padding=0,
                   ofm_base=0x60)


@cocotb.test()
async def test_conv_5x5_k3x3(dut):
    """K=36 forces 9 K-tiles. H=W=5, kh=kw=3, Cin=Cout=4, stride=1, padding=0.
    M=9, K=36, N=4. Raw INT32 low-byte output."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_5533)
    ifm = rng.integers(-4, 4, size=(5, 5, 4), dtype=np.int8)
    ker = rng.integers(-4, 4, size=(3, 3, 4, 4), dtype=np.int8)

    await run_conv(dut, ifm, ker, stride=1, padding=0,
                   ofm_base=0x80)


@cocotb.test()
async def test_conv_4x4_k3x3_pad1_relu_req(dut):
    """Padding=1: H'=W'=H=4 (same conv). M=16, K=36, N=4 + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_DAD1)
    ifm = rng.integers(-8, 8, size=(4, 4, 4), dtype=np.int8)
    ker = rng.integers(-8, 8, size=(3, 3, 4, 4), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 20

    await run_conv(dut, ifm, ker, stride=1, padding=1,
                   relu_en=True, req_en=True,
                   req_mult=req_mult, req_shift=req_shift,
                   ofm_base=0xA0)


@cocotb.test()
async def test_conv_5x5_k3x3_cout8_relu_req(dut):
    """N-tile: Cout=8 -> 2 N tiles. M=9, K=36, N=8. ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_C008)
    ifm = rng.integers(-8, 8, size=(5, 5, 4), dtype=np.int8)
    ker = rng.integers(-8, 8, size=(3, 3, 4, 8), dtype=np.int8)
    req_mult = 1 << 18
    req_shift = 22

    await run_conv(dut, ifm, ker, stride=1, padding=0,
                   relu_en=True, req_en=True,
                   req_mult=req_mult, req_shift=req_shift,
                   ofm_base=0xC0)


@cocotb.test()
async def test_conv_5x5_k3x3_cout8_bias_relu_req(dut):
    """Full pipeline: conv + per-N-tile bias + ReLU + global requantize."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_B1A5)
    ifm = rng.integers(-8, 8, size=(5, 5, 4), dtype=np.int8)
    ker = rng.integers(-8, 8, size=(3, 3, 4, 8), dtype=np.int8)
    bias = [int(v) for v in rng.integers(-200, 200, size=8, dtype=np.int32)]
    req_mult = 1 << 18
    req_shift = 22

    await run_conv(dut, ifm, ker, stride=1, padding=0,
                   relu_en=True, req_en=True,
                   req_mult=req_mult, req_shift=req_shift,
                   bias=bias,
                   ofm_base=0xE0,
                   bias_base=0x20)


@cocotb.test()
async def test_conv_5x5_k3x3_cout8_per_channel(dut):
    """Per-channel requantize across N=8 (one mult/shift per Cout lane)."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = np.random.default_rng(0xC0_DCC0)
    ifm = rng.integers(-8, 8, size=(5, 5, 4), dtype=np.int8)
    ker = rng.integers(-8, 8, size=(3, 3, 4, 8), dtype=np.int8)
    mults = [1 << 18, 1 << 19, -(1 << 18), 1 << 17,
             1 << 20, -(1 << 19), 1 << 18, 1 << 16]
    shifts = [12, 14, 11, 10, 15, 13, 12, 9]

    await run_conv(dut, ifm, ker, stride=1, padding=0,
                   req_en=True, per_channel=True,
                   mults=mults, shifts=shifts,
                   ofm_base=0x40)
