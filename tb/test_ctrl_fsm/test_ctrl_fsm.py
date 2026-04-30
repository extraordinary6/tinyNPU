"""Cocotb tests for rtl/ctrl_fsm.sv (top-level orchestration FSM).

State machine: IDLE -> LOAD_W -> COMPUTE -> WRITEBACK -> DONE -> IDLE
              (or IDLE -> ERR -> IDLE on M/N/K==0).
Sub-module starts are 1-cycle pulses on entry into LOAD_W / COMPUTE / WRITEBACK.
"""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


CLK_NS = 10
SETTLE_NS = 1


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.m_count.value = 0
    dut.n_count.value = 0
    dut.k_count.value = 0
    dut.wl_done.value = 0
    dut.if_done.value = 0
    dut.ow_done.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def pulse_start(dut, m=4, n=4, k=4):
    dut.m_count.value = m
    dut.n_count.value = n
    dut.k_count.value = k
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")


async def pulse_done(dut, signal):
    """Hold a *_done signal for one cycle (matches sub-module DONE pulse)."""
    signal.value = 1
    await RisingEdge(dut.clk)
    signal.value = 0
    await Timer(SETTLE_NS, units="ns")


@cocotb.test()
async def test_fsm_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert int(dut.err.value) == 0
    assert int(dut.wl_start.value) == 0
    assert int(dut.if_start.value) == 0
    assert int(dut.ow_start.value) == 0


@cocotb.test()
async def test_fsm_happy_path(dut):
    """Full IDLE->LOAD_W->COMPUTE->WRITEBACK->DONE->IDLE sequence."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=4)
    # After start pulse settles, state should be LOAD_W and wl_start high.
    assert int(dut.busy.value) == 1
    assert int(dut.wl_start.value) == 1
    assert int(dut.if_start.value) == 0
    assert int(dut.ow_start.value) == 0

    # Next cycle: wl_start drops back to 0 (only first cycle of state).
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.wl_start.value) == 0

    # Sub-module replies wl_done — FSM transitions to COMPUTE.
    await pulse_done(dut, dut.wl_done)
    assert int(dut.if_start.value) == 1
    assert int(dut.busy.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.if_start.value) == 0

    # if_done -> WRITEBACK
    await pulse_done(dut, dut.if_done)
    assert int(dut.ow_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.ow_start.value) == 0

    # ow_done -> DONE
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1
    assert int(dut.busy.value) == 1
    assert int(dut.err.value) == 0

    # next cycle back to IDLE
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0


@cocotb.test()
async def test_fsm_err_on_zero_dim(dut):
    """M==0 (or N==0 or K==0) on start makes the FSM jump straight to ERR."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    for trial in [(0, 4, 4), (4, 0, 4), (4, 4, 0)]:
        await reset(dut)
        m, n, k = trial
        await pulse_start(dut, m=m, n=n, k=k)
        # state == ERR right after the start pulse.
        assert int(dut.err.value) == 1, f"trial {trial}: err={int(dut.err.value)}"
        assert int(dut.busy.value) == 1
        assert int(dut.wl_start.value) == 0  # didn't enter LOAD_W
        # ERR -> IDLE next cycle.
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert int(dut.err.value) == 0
        assert int(dut.busy.value) == 0


@cocotb.test()
async def test_fsm_start_ignored_when_busy(dut):
    """Start pulse during a run should not perturb the in-flight FSM."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    await pulse_start(dut, m=4, n=4, k=4)
    assert int(dut.wl_start.value) == 1

    # Re-pulse start while in LOAD_W; FSM ignores start outside IDLE.
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await Timer(SETTLE_NS, units="ns")
    # Should still be in LOAD_W (not transitioning back to LOAD_W -> wl_start would re-pulse).
    assert int(dut.wl_start.value) == 0
    assert int(dut.busy.value) == 1

    # Drive the run to completion.
    await pulse_done(dut, dut.wl_done)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1


@cocotb.test()
async def test_fsm_back_to_back_runs(dut):
    """Two complete kicks in sequence."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    for _ in range(2):
        await pulse_start(dut, m=2, n=8, k=4)
        assert int(dut.wl_start.value) == 1
        await RisingEdge(dut.clk)
        await pulse_done(dut, dut.wl_done)
        await RisingEdge(dut.clk)
        await pulse_done(dut, dut.if_done)
        await RisingEdge(dut.clk)
        await pulse_done(dut, dut.ow_done)
        assert int(dut.done.value) == 1
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert int(dut.busy.value) == 0
