"""Cocotb tests for rtl/ctrl_fsm.sv (top-level orchestration FSM).

Sequence: IDLE -> LOAD_W -> [LOAD_BIAS] -> [LOAD_REQ] -> COMPUTE
                  -> (next tile? LOAD_W : WRITEBACK) -> DONE -> IDLE
                  (or IDLE -> ERR -> IDLE on M/N/K==0).
LOAD_BIAS / LOAD_REQ run only on the FIRST tile (when bias_en / pch_req_en).
COMPUTE loops back to LOAD_W until tile_idx == k_tiles_total - 1 (last_tile).
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
    dut.k_tiles_total.value = 0
    dut.bias_en.value = 0
    dut.pch_req_en.value = 0
    dut.wl_done.value = 0
    dut.bl_done.value = 0
    dut.rp_done.value = 0
    dut.if_done.value = 0
    dut.ow_done.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def pulse_start(dut, *, m=4, n=4, k=4, k_tiles=1, bias=0, pch=0):
    dut.m_count.value = m
    dut.n_count.value = n
    dut.k_count.value = k
    dut.k_tiles_total.value = k_tiles
    dut.bias_en.value = bias
    dut.pch_req_en.value = pch
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


async def run_compute_tile(dut):
    """Drive one COMPUTE tile to completion (assumes if_start has fired)."""
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)


@cocotb.test()
async def test_fsm_idle_after_reset(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert int(dut.err.value) == 0
    assert int(dut.wl_start.value) == 0
    assert int(dut.bl_start.value) == 0
    assert int(dut.rp_start.value) == 0
    assert int(dut.if_start.value) == 0
    assert int(dut.ow_start.value) == 0
    assert int(dut.tile_idx.value) == 0


@cocotb.test()
async def test_fsm_happy_path(dut):
    """K_TILES=1: IDLE->LOAD_W->COMPUTE->WRITEBACK->DONE->IDLE."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=4, k_tiles=1)
    assert int(dut.busy.value) == 1
    assert int(dut.wl_start.value) == 1
    assert int(dut.first_tile.value) == 1
    assert int(dut.last_tile.value) == 1
    assert int(dut.tile_idx.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.if_start.value) == 1
    assert int(dut.bl_start.value) == 0
    assert int(dut.rp_start.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    assert int(dut.ow_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0


@cocotb.test()
async def test_fsm_bias_path(dut):
    """bias_en=1, K_TILES=1: LOAD_W -> LOAD_BIAS -> COMPUTE."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=4, k_tiles=1, bias=1)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.bl_start.value) == 1
    assert int(dut.rp_start.value) == 0
    assert int(dut.if_start.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.bl_done)
    assert int(dut.if_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1


@cocotb.test()
async def test_fsm_bias_plus_pch_path(dut):
    """bias_en=1, pch_req_en=1, K_TILES=1: LOAD_W -> LOAD_BIAS -> LOAD_REQ -> COMPUTE."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=4, k_tiles=1, bias=1, pch=1)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.bl_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.bl_done)
    assert int(dut.rp_start.value) == 1
    assert int(dut.if_start.value) == 0

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.rp_done)
    assert int(dut.if_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1


@cocotb.test()
async def test_fsm_per_channel_path(dut):
    """pch_req_en=1, bias_en=0, K_TILES=1: LOAD_W -> LOAD_REQ -> COMPUTE."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=4, k_tiles=1, pch=1)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.bl_start.value) == 0
    assert int(dut.rp_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.rp_done)
    assert int(dut.if_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1


@cocotb.test()
async def test_fsm_ktile_loop(dut):
    """K_TILES=3: LOAD_W -> COMPUTE three times, only last triggers WRITEBACK."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    K_TILES = 3
    await pulse_start(dut, m=4, n=4, k=12, k_tiles=K_TILES)

    for tile in range(K_TILES):
        # In LOAD_W of tile `tile`. tile_idx should be `tile`.
        assert int(dut.wl_start.value) == 1, f"tile={tile}: wl_start"
        assert int(dut.tile_idx.value) == tile, f"tile={tile}: tile_idx={int(dut.tile_idx.value)}"
        assert int(dut.first_tile.value) == (1 if tile == 0 else 0)
        assert int(dut.last_tile.value) == (1 if tile == K_TILES - 1 else 0)

        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        await pulse_done(dut, dut.wl_done)
        # First tile only could go to LOAD_BIAS / LOAD_REQ; here both en bits are 0
        # so we should be in COMPUTE directly.
        assert int(dut.if_start.value) == 1, f"tile={tile}: if_start after wl_done"

        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        await pulse_done(dut, dut.if_done)
        if tile < K_TILES - 1:
            # Loops back to LOAD_W of next tile.
            assert int(dut.wl_start.value) == 1, f"tile={tile}: re-LOAD_W"
            assert int(dut.ow_start.value) == 0, f"tile={tile}: must not start ofm"
        else:
            # Last tile: ofm_writer starts.
            assert int(dut.ow_start.value) == 1, "last tile: ow_start"

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.busy.value) == 0


@cocotb.test()
async def test_fsm_ktile_with_bias_first_only(dut):
    """K_TILES=2, bias_en=1: LOAD_BIAS only on first tile, not on second."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    await pulse_start(dut, m=4, n=4, k=8, k_tiles=2, bias=1)

    # Tile 0: LOAD_W -> LOAD_BIAS -> COMPUTE.
    assert int(dut.wl_start.value) == 1
    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.bl_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.bl_done)
    assert int(dut.if_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)

    # Tile 1: LOAD_W -> COMPUTE directly (no LOAD_BIAS).
    assert int(dut.wl_start.value) == 1
    assert int(dut.first_tile.value) == 0
    assert int(dut.last_tile.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.wl_done)
    assert int(dut.bl_start.value) == 0, "bias should not reload on second tile"
    assert int(dut.if_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.if_done)
    assert int(dut.ow_start.value) == 1

    await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")
    await pulse_done(dut, dut.ow_done)
    assert int(dut.done.value) == 1


@cocotb.test()
async def test_fsm_err_on_zero_dim(dut):
    """M==0 (or N==0 or K==0 or k_tiles_total==0) on start makes the FSM jump to ERR."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())

    for trial in [(0, 4, 4, 1), (4, 0, 4, 1), (4, 4, 0, 1), (4, 4, 4, 0)]:
        await reset(dut)
        m, n, k, kt = trial
        await pulse_start(dut, m=m, n=n, k=k, k_tiles=kt)
        assert int(dut.err.value) == 1, f"trial {trial}: err={int(dut.err.value)}"
        assert int(dut.busy.value) == 1
        assert int(dut.wl_start.value) == 0
        await RisingEdge(dut.clk)
        await Timer(SETTLE_NS, units="ns")
        assert int(dut.err.value) == 0
        assert int(dut.busy.value) == 0


@cocotb.test()
async def test_fsm_back_to_back_runs(dut):
    """Two complete K_TILES=1 kicks in sequence."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    await reset(dut)

    for _ in range(2):
        await pulse_start(dut, m=2, n=8, k=4, k_tiles=1)
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
