"""Cocotb tests for rtl/weight_loader.sv (drives a sram_wrapper via wl_harness)."""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ROWS = 4
COLS = 4
A_W = 8
CLK_NS = 10
SETTLE_NS = 1


def pack_tile(tile):
    val = 0
    for r in range(ROWS):
        for c in range(COLS):
            byte = int(tile[r][c]) & 0xFF
            val |= byte << ((r * COLS + c) * A_W)
    return val


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.bd_we.value = 0
    dut.bd_addr.value = 0
    dut.bd_wdata.value = 0
    dut.start.value = 0
    dut.base_addr.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def backdoor_store(dut, addr, packed):
    dut.bd_we.value = 1
    dut.bd_addr.value = addr
    dut.bd_wdata.value = packed
    await RisingEdge(dut.clk)
    dut.bd_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def issue_load(dut, base_addr):
    dut.start.value = 1
    dut.base_addr.value = base_addr
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_wl_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert int(dut.w_load.value) == 0


@cocotb.test()
async def test_wl_single_load(dut):
    """start -> 3 cycles later, a single w_load pulse with the SRAM contents."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    tile = [[1, 2, 3, 4],
            [-1, -2, -3, -4],
            [127, -128, 0, 50],
            [-50, 100, -100, 25]]
    packed = pack_tile(tile)

    await backdoor_store(dut, addr=0x10, packed=packed)
    await issue_load(dut, base_addr=0x10)
    # FSM after issue_load: IDLE->FETCH transition just happened, so state==FETCH now.
    assert int(dut.w_load.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    # Now state==LOAD: w_load high, w_out has SRAM data.
    assert int(dut.w_load.value) == 1, f"w_load={int(dut.w_load.value)}"
    assert int(dut.w_out.value) == packed

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    # state==DONE: done=1, w_load=0.
    assert int(dut.done.value) == 1
    assert int(dut.w_load.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    # back to IDLE
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0


@cocotb.test()
async def test_wl_back_to_back(dut):
    """Two distinct tiles at different addresses, loaded in sequence."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0x77)
    addrs = [0x05, 0x42]
    tiles = []
    for a in addrs:
        tile = [[rng.randint(-128, 127) for _ in range(COLS)] for _ in range(ROWS)]
        tiles.append(tile)
        await backdoor_store(dut, addr=a, packed=pack_tile(tile))

    for a, t in zip(addrs, tiles):
        await issue_load(dut, base_addr=a)
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert int(dut.w_load.value) == 1
        assert int(dut.w_out.value) == pack_tile(t)
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_wl_zero_tile(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await backdoor_store(dut, addr=0, packed=0)
    await issue_load(dut, base_addr=0)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.w_load.value) == 1
    assert int(dut.w_out.value) == 0
