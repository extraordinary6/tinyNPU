"""Cocotb tests for rtl/ifm_feeder.sv (M-row staggered IFM streamer).

Expected output schedule (cocotb cycle T after issue_start, T=0 = first cycle in S_READ):
    a_out[r] @ cycle (i + r + 1) == A[i, r]   for 0 <= i < M
    a_out[r] is 0 outside its valid window (drain pads with zeros).
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ROWS = 4
A_W = 8
A_BUS_BITS = ROWS * A_W
CLK_NS = 10
SETTLE_NS = 1


def s8(x: int) -> int:
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def pack_row(row):
    val = 0
    for r, v in enumerate(row):
        val |= (int(v) & 0xFF) << (r * A_W)
    return val


def unpack_a_out(raw: int):
    return [s8((raw >> (r * A_W)) & 0xFF) for r in range(ROWS)]


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.bd_we.value = 0
    dut.bd_addr.value = 0
    dut.bd_wdata.value = 0
    dut.start.value = 0
    dut.m_count.value = 0
    dut.base_addr.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def backdoor_store(dut, rows, base):
    """rows: list of M rows, each row is list of ROWS int8."""
    for i, row in enumerate(rows):
        dut.bd_we.value = 1
        dut.bd_addr.value = base + i
        dut.bd_wdata.value = pack_row(row)
        await RisingEdge(dut.clk)
    dut.bd_we.value = 0
    await Timer(SETTLE_NS, units="ns")


async def issue_start(dut, m_count, base_addr):
    dut.start.value = 1
    dut.m_count.value = m_count
    dut.base_addr.value = base_addr
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")


async def run_and_capture(dut, A, base):
    """Pre-load A, issue start, then sample a_out for M+ROWS cycles."""
    M = len(A)
    await backdoor_store(dut, A, base)
    await issue_start(dut, m_count=M, base_addr=base)
    # After issue_start, settle: state == S_READ, this is "cycle 0".
    # But cycle 0's a_out is from pipe[0]=sram_rdata (still X) gated to 0,
    # and pipe[1..3]=0. So a_out=0 at cycle 0.
    samples = []
    samples.append(unpack_a_out(int(dut.a_out.value)))  # cycle 0
    n_cycles = M + ROWS  # need cycles 1..M+ROWS-1 to capture all valid lanes
    for _ in range(n_cycles):
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        samples.append(unpack_a_out(int(dut.a_out.value)))
    return samples  # samples[t] corresponds to cycle t


def expected_at(A, t, r):
    """Expected a_out[r] at cycle t. Returns 0 if outside valid window."""
    M = len(A)
    i = t - r - 1
    if 0 <= i < M:
        return s8(A[i][r])
    return 0


@cocotb.test()
async def test_if_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert unpack_a_out(int(dut.a_out.value)) == [0] * ROWS


@cocotb.test()
async def test_if_M_equals_rows(dut):
    """M=4 (square): full stagger pattern, lanes valid at the expected cycles."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [-1, -2, -3, -4]]
    samples = await run_and_capture(dut, A, base=0)

    for t, vec in enumerate(samples):
        for r in range(ROWS):
            exp = expected_at(A, t, r)
            assert vec[r] == exp, f"cycle {t} lane {r}: got {vec[r]} expected {exp}"


@cocotb.test()
async def test_if_M_smaller(dut):
    """M=2 (less than ROWS): valid windows are short, outside is 0."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    A = [[10, 20, 30, 40],
         [-10, -20, -30, -40]]
    samples = await run_and_capture(dut, A, base=5)

    for t, vec in enumerate(samples):
        for r in range(ROWS):
            exp = expected_at(A, t, r)
            assert vec[r] == exp, f"cycle {t} lane {r}: got {vec[r]} expected {exp}"


@cocotb.test()
async def test_if_M_larger(dut):
    """M=8 (more than ROWS): long stream, lane 0 stays valid for M cycles."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    rng = random.Random(0x1F)
    M = 8
    A = [[rng.randint(-128, 127) for _ in range(ROWS)] for _ in range(M)]
    samples = await run_and_capture(dut, A, base=0x10)

    for t, vec in enumerate(samples):
        for r in range(ROWS):
            exp = expected_at(A, t, r)
            assert vec[r] == exp, f"cycle {t} lane {r}: got {vec[r]} expected {exp}"


@cocotb.test()
async def test_if_done_pulse(dut):
    """done rises exactly once after the drain finishes."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    A = [[1, 2, 3, 4]] * 4
    await backdoor_store(dut, A, base=0)
    await issue_start(dut, m_count=4, base_addr=0)
    seen_done = 0
    # State sequence: READ (4 cycles) -> DRAIN (ROWS=4 cycles) -> DONE (1 cycle) -> IDLE
    # done=1 only in DONE state. Total cycles to observe = M + ROWS + 2.
    for _ in range(4 + ROWS + 2):
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        if int(dut.done.value):
            seen_done += 1
    assert seen_done == 1, f"done pulsed {seen_done} times (expected 1)"
