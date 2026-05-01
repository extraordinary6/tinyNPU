"""Cocotb tests for rtl/req_param_loader.sv (drives a sram_wrapper via rp_harness).

Sequence:
  IDLE -> FETCH_MULT (sram_en=1, addr=mult_base) -> LATCH_MULT (rdata valid)
       -> FETCH_SHIFT (addr=shift_base)          -> LATCH_SHIFT
       -> DONE -> IDLE
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


def pack_mults(mults):
    val = 0
    for c, m in enumerate(mults):
        val |= (int(m) & 0xFFFF_FFFF) << (c * P_W)
    return val


def pack_shift_word(shifts):
    """Place each shift in the low byte of its lane slot inside an LANES*P_W-bit word.

    Software writes one INT32 array of shifts so the layout matches the mult word
    (low 6 bits used, upper bits ignored).
    """
    val = 0
    for c, s in enumerate(shifts):
        val |= (int(s) & 0xFF) << (c * P_W)
    return val


def unpack_mults(raw):
    out = []
    for c in range(LANES):
        v = (raw >> (c * P_W)) & 0xFFFF_FFFF
        if v & (1 << (P_W - 1)):
            v -= 1 << P_W
        out.append(v)
    return out


def unpack_shifts(raw):
    return [(raw >> (c * 6)) & 0x3F for c in range(LANES)]


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.bd_we.value = 0
    dut.bd_addr.value = 0
    dut.bd_wdata.value = 0
    dut.start.value = 0
    dut.mult_base_addr.value = 0
    dut.shift_base_addr.value = 0
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


async def issue_load(dut, mult_addr, shift_addr):
    dut.start.value = 1
    dut.mult_base_addr.value = mult_addr
    dut.shift_base_addr.value = shift_addr
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")


async def wait_done(dut, max_cycles=20):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.done.value):
            return
    raise TimeoutError("req_param_loader did not signal done")


@cocotb.test()
async def test_rp_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0


@cocotb.test()
async def test_rp_load(dut):
    """Load mult and shift words from two addresses, latch into outputs."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    mults = [12345, -67890, 1, -1]
    shifts = [0, 5, 31, 16]
    mult_addr = 0x12
    shift_addr = 0x34
    await backdoor_store(dut, addr=mult_addr, packed=pack_mults(mults))
    await backdoor_store(dut, addr=shift_addr, packed=pack_shift_word(shifts))

    await issue_load(dut, mult_addr=mult_addr, shift_addr=shift_addr)
    await wait_done(dut)

    assert unpack_mults(int(dut.mult_out.value)) == mults
    assert unpack_shifts(int(dut.shift_out.value)) == shifts

    # Returns to IDLE next cycle.
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0


@cocotb.test()
async def test_rp_back_to_back(dut):
    """Two distinct (mult, shift) sets at different addresses."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0xBEEF)
    sets = []
    for trial in range(2):
        mults = [rng.randint(-(1 << 30), 1 << 30) for _ in range(LANES)]
        shifts = [rng.randint(0, 31) for _ in range(LANES)]
        m_addr = 0x40 + trial * 2
        s_addr = 0x40 + trial * 2 + 1
        await backdoor_store(dut, addr=m_addr, packed=pack_mults(mults))
        await backdoor_store(dut, addr=s_addr, packed=pack_shift_word(shifts))
        sets.append((m_addr, s_addr, mults, shifts))

    for m_addr, s_addr, mults, shifts in sets:
        await issue_load(dut, mult_addr=m_addr, shift_addr=s_addr)
        await wait_done(dut)
        assert unpack_mults(int(dut.mult_out.value)) == mults
        assert unpack_shifts(int(dut.shift_out.value)) == shifts
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_rp_zero_params(dut):
    """All-zero mult and shift words."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await backdoor_store(dut, addr=0, packed=0)
    await backdoor_store(dut, addr=1, packed=0)
    await issue_load(dut, mult_addr=0, shift_addr=1)
    await wait_done(dut)
    assert unpack_mults(int(dut.mult_out.value)) == [0] * LANES
    assert unpack_shifts(int(dut.shift_out.value)) == [0] * LANES
