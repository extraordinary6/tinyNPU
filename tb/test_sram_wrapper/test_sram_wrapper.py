"""Cocotb tests for rtl/sram_wrapper.sv (behavioral synchronous SRAM)."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


CLK_NS = 10
SETTLE_NS = 1


async def reset(dut, cycles=2):
    dut.en.value = 0
    dut.we.value = 0
    dut.addr.value = 0
    dut.wdata.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def write(dut, addr, data):
    dut.en.value = 1
    dut.we.value = 1
    dut.addr.value = addr
    dut.wdata.value = data
    await RisingEdge(dut.clk)
    dut.en.value = 0
    dut.we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def read(dut, addr):
    dut.en.value = 1
    dut.we.value = 0
    dut.addr.value = addr
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(SETTLE_NS, units="ns")
    return int(dut.rdata.value)


@cocotb.test()
async def test_sram_basic_rw(dut):
    """Write then read returns the same data, with 1-cycle latency."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    pairs = [(0, 0xDEADBEEF), (5, 0x12345678), (255, 0xABCD0001)]
    for addr, data in pairs:
        await write(dut, addr, data)
    for addr, data in pairs:
        got = await read(dut, addr)
        assert got == data, f"addr={addr}: got=0x{got:08x} expected=0x{data:08x}"


@cocotb.test()
async def test_sram_overwrite(dut):
    """Re-writing an address replaces the old value."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await write(dut, 17, 0xAAAA_AAAA)
    assert (await read(dut, 17)) == 0xAAAA_AAAA
    await write(dut, 17, 0x5555_5555)
    assert (await read(dut, 17)) == 0x5555_5555


@cocotb.test()
async def test_sram_en_low_holds(dut):
    """en=0 leaves both memory and rdata unchanged."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await write(dut, 3, 0xCAFEBABE)
    val0 = await read(dut, 3)

    # Drive bogus addr/wdata with en=0 — must not write.
    dut.en.value = 0
    dut.we.value = 1
    dut.addr.value = 3
    dut.wdata.value = 0xDEADBEEF
    for _ in range(3):
        await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")

    # rdata still has the previous read value.
    assert int(dut.rdata.value) == val0

    val1 = await read(dut, 3)
    assert val1 == 0xCAFEBABE, f"data corrupted under en=0: got 0x{val1:08x}"


@cocotb.test()
async def test_sram_random(dut):
    """Random fuzz: 200 mixed reads/writes against a Python dict reference."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0x5AAA)
    ref = {}
    for _ in range(200):
        addr = rng.randint(0, 255)
        if rng.random() < 0.5:
            data = rng.randint(0, 0xFFFF_FFFF)
            await write(dut, addr, data)
            ref[addr] = data
        elif addr in ref:
            got = await read(dut, addr)
            assert got == ref[addr], f"addr={addr} got=0x{got:08x} ref=0x{ref[addr]:08x}"
