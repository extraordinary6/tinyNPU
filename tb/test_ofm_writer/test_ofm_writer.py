"""Cocotb tests for rtl/ofm_writer.sv (drives sram_wrapper via ow_harness)."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
O_W = 8
D_BUS_BITS = LANES * O_W
CLK_NS = 10
SETTLE_NS = 1


def pack_word(lane_bytes):
    val = 0
    for c, v in enumerate(lane_bytes):
        val |= (int(v) & 0xFF) << (c * O_W)
    return val


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.m_count.value = 0
    dut.base_addr.value = 0
    dut.data_in.value = 0
    dut.data_valid.value = 0
    dut.bd_re.value = 0
    dut.bd_addr.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def backdoor_read(dut, addr):
    dut.bd_re.value = 1
    dut.bd_addr.value = addr
    await RisingEdge(dut.clk)
    dut.bd_re.value = 0
    await Timer(SETTLE_NS, units="ns")
    return int(dut.bd_rdata.value)


async def issue_start(dut, m_count, base_addr):
    dut.start.value = 1
    dut.m_count.value = m_count
    dut.base_addr.value = base_addr
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")


async def feed_word(dut, packed, valid=1):
    dut.data_in.value = packed
    dut.data_valid.value = valid
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_ow_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0


@cocotb.test()
async def test_ow_contiguous(dut):
    """data_valid=1 every cycle: M words written contiguously from base_addr."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    M = 6
    base = 0x20
    payload = [[i, 2*i, -i, 100 - i] for i in range(M)]

    await issue_start(dut, m_count=M, base_addr=base)
    for w in payload:
        await feed_word(dut, pack_word(w), valid=1)
    # The last feed_word landed the M-th beat; state == S_DONE this cycle.
    assert int(dut.done.value) == 1
    dut.data_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0

    # Read back via backdoor.
    for i, w in enumerate(payload):
        got = await backdoor_read(dut, base + i)
        assert got == pack_word(w), (
            f"addr={base+i}: got=0x{got:08x} expected=0x{pack_word(w):08x}"
        )


@cocotb.test()
async def test_ow_with_bubbles(dut):
    """data_valid pulses with gaps: only valid cycles consume a write slot."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    M = 4
    base = 0x40
    payload = [[10, 20, 30, 40], [50, 60, 70, 80],
               [-1, -2, -3, -4], [127, 0, -128, 1]]

    await issue_start(dut, m_count=M, base_addr=base)

    # Pattern: V _ V V _ _ V  (V=valid, _=bubble), need 4 valid total.
    pattern = [1, 0, 1, 1, 0, 0, 1]
    idx = 0
    for v in pattern:
        if v and idx < M:
            await feed_word(dut, pack_word(payload[idx]), valid=1)
            idx += 1
        else:
            await feed_word(dut, 0xDEADBEEF, valid=0)

    dut.data_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")

    for i, w in enumerate(payload):
        got = await backdoor_read(dut, base + i)
        assert got == pack_word(w), f"addr={base+i}: bubble run mismatch"


@cocotb.test()
async def test_ow_random(dut):
    """Random sizes + random valid masks."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    rng = random.Random(0xFACE)
    for trial in range(3):
        await reset(dut)
        M = rng.randint(1, 12)
        base = rng.randint(0, 100)
        payload = [[rng.randint(-128, 127) for _ in range(LANES)] for _ in range(M)]

        await issue_start(dut, m_count=M, base_addr=base)

        idx = 0
        steps = 0
        while idx < M and steps < 200:
            steps += 1
            if rng.random() < 0.7:
                await feed_word(dut, pack_word(payload[idx]), valid=1)
                idx += 1
            else:
                await feed_word(dut, 0, valid=0)
        assert idx == M

        dut.data_valid.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")

        for i, w in enumerate(payload):
            got = await backdoor_read(dut, base + i)
            assert got == pack_word(w), f"trial={trial} idx={i}"
