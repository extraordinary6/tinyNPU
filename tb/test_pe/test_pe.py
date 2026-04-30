"""Cocotb tests for rtl/pe.sv (weight-stationary MAC cell).

Driver timing convention used below:
  1. Set inputs (a_in, psum_in, w_load, w_in) via dut.X.value = ...
  2. await RisingEdge(dut.clk)         <- the latching edge
  3. await Timer(SETTLE_NS)            <- let NBA region settle
  4. Read outputs (a_out, psum_out)    <- reflect the inputs from step 1

So after one RisingEdge, a_out == a_in_prev_cycle and
psum_out == psum_in_prev_cycle + a_in_prev_cycle * w_q_prev_cycle.

All quantities are interpreted as two's-complement signed.
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


CLK_PERIOD_NS = 10
SETTLE_NS = 1


def s8(x: int) -> int:
    """Wrap a Python int into a signed 8-bit value."""
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def s32(x: int) -> int:
    """Wrap a Python int into a signed 32-bit value."""
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x & (1 << 31) else x


def read_s8(sig) -> int:
    return s8(int(sig.value))


def read_s32(sig) -> int:
    return s32(int(sig.value))


async def reset(dut, cycles: int = 2) -> None:
    dut.rst_n.value = 0
    dut.w_load.value = 0
    dut.w_in.value = 0
    dut.a_in.value = 0
    dut.psum_in.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def load_weight(dut, w: int) -> None:
    """Latch a weight; weight is visible to the multiplier on the NEXT cycle."""
    dut.w_load.value = 1
    dut.w_in.value = w
    await RisingEdge(dut.clk)
    dut.w_load.value = 0
    dut.w_in.value = 0
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_pe_reset(dut):
    """After rst_n, both pipeline outputs are zero."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)
    assert read_s8(dut.a_out) == 0
    assert read_s32(dut.psum_out) == 0


@cocotb.test()
async def test_pe_a_passthrough(dut):
    """a_in flows through the a_q pipeline register to a_out."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    samples = [5, -7, 127, -128, 0, 42, -1]
    for x in samples:
        dut.a_in.value = x
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        got = read_s8(dut.a_out)
        assert got == s8(x), f"a_out: got {got}, expected {s8(x)}"


@cocotb.test()
async def test_pe_weight_load(dut):
    """Loaded weight participates in the very next MAC."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    await load_weight(dut, 5)

    # First MAC after the load: psum_in=0, a_in=3 -> next-cycle psum_out = 0 + 3*5 = 15.
    dut.a_in.value = 3
    dut.psum_in.value = 0
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert read_s32(dut.psum_out) == 15

    # Reload to a new weight, verify it takes effect immediately.
    await load_weight(dut, -4)
    dut.a_in.value = 6
    dut.psum_in.value = 100
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert read_s32(dut.psum_out) == 100 + 6 * (-4)  # = 76


@cocotb.test()
async def test_pe_mac_signed(dut):
    """Signed multiply across the four sign quadrants and INT8 extremes."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    cases = [
        # (w, a, psum_in, expected_psum_out)
        (  7,   6,    0,        42),
        ( -7,   6,    0,       -42),
        (  7,  -6,    0,       -42),
        ( -7,  -6,    0,        42),
        (-128, -128,  0,    16384),  # 2^14
        ( 127,  127,  0,    16129),
        (-128,  127,  0,   -16256),
        (  1,    1,  -1,         0),
        (  3,    4,  100,      112),
    ]

    for w, a, p_in, expected in cases:
        await reset(dut)
        await load_weight(dut, w)
        dut.a_in.value = a
        dut.psum_in.value = p_in
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        got = read_s32(dut.psum_out)
        assert got == expected, (
            f"w={w} a={a} psum_in={p_in}: got {got}, expected {expected}"
        )


@cocotb.test()
async def test_pe_streaming_random(dut):
    """Stream random inputs; check the MAC against the model each cycle."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0xC0C07B)
    w = rng.randint(-128, 127)
    await load_weight(dut, w)

    n = 64
    a_seq = [rng.randint(-128, 127) for _ in range(n)]
    p_seq = [rng.randint(-1000, 1000) for _ in range(n)]

    for a, p in zip(a_seq, p_seq):
        dut.a_in.value = a
        dut.psum_in.value = p
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")

        exp = s32(p + a * w)
        got = read_s32(dut.psum_out)
        assert got == exp, f"a={a} p={p} w={w}: got {got}, expected {exp}"


@cocotb.test()
async def test_pe_weight_hold(dut):
    """w_q holds its value across many cycles when w_load stays low."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())
    await reset(dut)

    await load_weight(dut, 9)

    for k in range(8):
        a = k - 3  # -3..4
        dut.a_in.value = a
        dut.psum_in.value = 0
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        exp = a * 9
        got = read_s32(dut.psum_out)
        assert got == exp, f"k={k} a={a}: got {got}, expected {exp}"
