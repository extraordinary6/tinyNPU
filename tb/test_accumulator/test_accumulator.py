"""Cocotb tests for rtl/accumulator.sv (per-lane INT32 accumulator).

Bus layout (LSB-first): psum_in / acc_out have lane c at bits [c*P_W +: P_W].
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
P_W = 32
CLK_NS = 10
SETTLE_NS = 1


def s32(x: int) -> int:
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x & (1 << 31) else x


def pack_psum(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0xFFFF_FFFF) << (c * P_W)
    return val


def unpack_acc(raw: int):
    return [s32((raw >> (c * P_W)) & 0xFFFF_FFFF) for c in range(LANES)]


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.clr.value = 0
    dut.en.value = 0
    dut.psum_in.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def step(dut, *, en_mask, psum):
    dut.en.value = en_mask
    dut.psum_in.value = pack_psum(psum)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_acc_reset(dut):
    """All lanes are zero after reset."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert unpack_acc(int(dut.acc_out.value)) == [0] * LANES


@cocotb.test()
async def test_acc_basic_sum(dut):
    """en=1111 accumulates each lane independently."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    seq = [
        [10, 20, 30, 40],
        [-5, 100, 0, 1],
        [1, 1, -1000, 1],
    ]
    expected = [0] * LANES
    for vec in seq:
        await step(dut, en_mask=0b1111, psum=vec)
        for c in range(LANES):
            expected[c] += vec[c]
        got = unpack_acc(int(dut.acc_out.value))
        assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def test_acc_clear(dut):
    """clr resets all lanes synchronously, regardless of en."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await step(dut, en_mask=0b1111, psum=[7, 8, 9, 10])
    await step(dut, en_mask=0b1111, psum=[1, 1, 1, 1])
    assert unpack_acc(int(dut.acc_out.value)) == [8, 9, 10, 11]

    # Assert clr alongside en — clr must win.
    dut.clr.value = 1
    await step(dut, en_mask=0b1111, psum=[100, 100, 100, 100])
    dut.clr.value = 0
    assert unpack_acc(int(dut.acc_out.value)) == [0] * LANES


@cocotb.test()
async def test_acc_per_lane_enable(dut):
    """en[c]=0 freezes lane c while others accumulate."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await step(dut, en_mask=0b1111, psum=[1, 2, 3, 4])
    # Only enable lanes 0 and 2.
    await step(dut, en_mask=0b0101, psum=[10, 20, 30, 40])
    expected = [1 + 10, 2, 3 + 30, 4]
    assert unpack_acc(int(dut.acc_out.value)) == expected


@cocotb.test()
async def test_acc_random(dut):
    """Random per-cycle en mask + psum, compared lane-by-lane to a model."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0xACC)
    expected = [0] * LANES
    for _ in range(64):
        en_mask = rng.randint(0, 15)
        psum = [rng.randint(-1_000_000, 1_000_000) for _ in range(LANES)]
        await step(dut, en_mask=en_mask, psum=psum)
        for c in range(LANES):
            if (en_mask >> c) & 1:
                expected[c] = s32(expected[c] + psum[c])
        got = unpack_acc(int(dut.acc_out.value))
        assert got == expected, f"en={en_mask:04b} psum={psum} got={got} expected={expected}"
