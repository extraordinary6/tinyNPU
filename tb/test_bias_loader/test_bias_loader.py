"""Cocotb tests for rtl/bias_loader.sv (drives a sram_wrapper via bl_harness).

Sequence:
  IDLE -> FETCH (sram_en=1, addr=base_addr)
       -> LATCH (rdata valid, bias_out updated)
       -> DONE  (one-cycle pulse)
       -> IDLE
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


def pack(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0xFFFF_FFFF) << (c * P_W)
    return val


def unpack(raw):
    return [s32((raw >> (c * P_W)) & 0xFFFF_FFFF) for c in range(LANES)]


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


async def wait_done(dut, max_cycles=20):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.done.value):
            return
    raise TimeoutError("bias_loader did not signal done")


@cocotb.test()
async def test_bl_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert int(dut.bias_out.value) == 0


@cocotb.test()
async def test_bl_single_load(dut):
    """Pre-load one bias vector, kick the loader, check latch + done pulse."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    biases = [12345, -67890, 0, -1]
    addr = 0x10
    await backdoor_store(dut, addr=addr, packed=pack(biases))
    await issue_load(dut, base_addr=addr)

    # FSM: state==FETCH right after issue_load (busy=1, done=0).
    assert int(dut.busy.value) == 1
    assert int(dut.done.value) == 0

    await wait_done(dut)
    assert unpack(int(dut.bias_out.value)) == biases

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0


@cocotb.test()
async def test_bl_back_to_back(dut):
    """Two distinct bias vectors at different addresses."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0xB1A5)
    addrs = [0x05, 0x42]
    sets = []
    for a in addrs:
        biases = [rng.randint(-(1 << 30), 1 << 30) for _ in range(LANES)]
        sets.append((a, biases))
        await backdoor_store(dut, addr=a, packed=pack(biases))

    for a, biases in sets:
        await issue_load(dut, base_addr=a)
        await wait_done(dut)
        assert unpack(int(dut.bias_out.value)) == biases
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_bl_zero(dut):
    """All-zero bias vector."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await backdoor_store(dut, addr=0, packed=0)
    await issue_load(dut, base_addr=0)
    await wait_done(dut)
    assert unpack(int(dut.bias_out.value)) == [0] * LANES


@cocotb.test()
async def test_bl_holds_after_done(dut):
    """bias_out holds its value after the kick completes (used by bias_relu downstream)."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    biases = [100, 200, -300, 400]
    await backdoor_store(dut, addr=0x20, packed=pack(biases))
    await issue_load(dut, base_addr=0x20)
    await wait_done(dut)
    assert unpack(int(dut.bias_out.value)) == biases

    # Idle for several cycles; bias_out must not change.
    for _ in range(5):
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert unpack(int(dut.bias_out.value)) == biases
