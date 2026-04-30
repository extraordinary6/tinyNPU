"""Smoke test for the cocotb + Icarus + py37 toolchain.

Validates: clock generation, sync reset, 1-cycle flip-flop latency.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset(dut, cycles=2):
    dut.rst_n.value = 0
    dut.d.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_dff_smoke(dut):
    """After reset, q captures d on each rising edge."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    for bit in [1, 0, 1, 1, 0, 0, 1, 0]:
        dut.d.value = bit
        await RisingEdge(dut.clk)
        await Timer(1, units="ns")
        got = int(dut.q.value)
        assert got == bit, f"d={bit} -> q={got}, expected {bit}"


@cocotb.test()
async def test_dff_reset(dut):
    """Asserting rst_n mid-operation forces q to 0."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    dut.d.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    assert int(dut.q.value) == 1

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, units="ns")
    assert int(dut.q.value) == 0, "rst_n should clear q"
