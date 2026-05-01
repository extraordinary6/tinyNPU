"""Cocotb tests for rtl/row_accumulator.sv (per-(row, lane) INT32 accumulator).

Behaviour each cycle:
  read_value = first_tile ? 0 : mem[row_idx]
  sum_value  = read_value + psum_in
  if data_valid: mem[row_idx] <= sum_value
  acc_out    = sum_value (combinational)
"""

from __future__ import annotations

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
P_W = 32
M_W = 16
M_MAX = 64
CLK_NS = 10
SETTLE_NS = 1


def s32(x):
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
    dut.data_valid.value = 0
    dut.row_idx.value = 0
    dut.first_tile.value = 0
    dut.psum_in.value = 0
    cocotb.start_soon(Clock(dut.clk, CLK_NS, units="ns").start())
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await Timer(SETTLE_NS, units="ns")


async def feed(dut, *, row, first_tile, psum):
    dut.data_valid.value = 1
    dut.row_idx.value = row
    dut.first_tile.value = first_tile
    dut.psum_in.value = pack(psum)
    await Timer(SETTLE_NS, units="ns")
    got = unpack(int(dut.acc_out.value))
    await RisingEdge(dut.clk)
    dut.data_valid.value = 0
    return got


@cocotb.test()
async def test_ra_first_tile_writethrough(dut):
    """first_tile=1 -> acc_out = psum_in (overwrites previous memory)."""
    await reset(dut)
    cases = [
        (0, [10, -20, 30, -40]),
        (1, [1, 2, 3, 4]),
        (2, [-100, 100, 50, -50]),
        (3, [0, 0, 0, 0]),
    ]
    for row, psum in cases:
        got = await feed(dut, row=row, first_tile=1, psum=psum)
        assert got == psum, f"row={row}: got={got} expected={psum}"


@cocotb.test()
async def test_ra_two_tiles(dut):
    """Tile 0 writethrough, tile 1 reads stored row + new psum, accumulates."""
    await reset(dut)

    M = 4
    tile0 = [[10, 20, 30, 40], [-1, -2, -3, -4], [100, -100, 50, -50], [7, 8, 9, 10]]
    tile1 = [[5, 5, 5, 5], [10, -10, 10, -10], [-50, 50, -50, 50], [3, 4, 5, 6]]

    # Tile 0 (first_tile=1).
    for row in range(M):
        got = await feed(dut, row=row, first_tile=1, psum=tile0[row])
        assert got == tile0[row]

    # Tile 1 (first_tile=0). acc_out should equal tile0 + tile1.
    for row in range(M):
        got = await feed(dut, row=row, first_tile=0, psum=tile1[row])
        expected = [tile0[row][c] + tile1[row][c] for c in range(LANES)]
        assert got == expected, f"row={row}: got={got} expected={expected}"


@cocotb.test()
async def test_ra_three_tiles_random(dut):
    """K_TILES=3 random fuzz: stored value should equal sum of all psum_in."""
    await reset(dut)

    rng = random.Random(0x10E_3)
    M = 8
    K_TILES = 3
    tiles = []
    for _ in range(K_TILES):
        tiles.append([[rng.randint(-(1 << 28), 1 << 28) for _ in range(LANES)] for _ in range(M)])

    for k in range(K_TILES):
        for row in range(M):
            got = await feed(dut, row=row, first_tile=(1 if k == 0 else 0), psum=tiles[k][row])
            expected = [sum(tiles[t][row][c] for t in range(k + 1)) for c in range(LANES)]
            assert got == expected, f"k={k} row={row}: got={got} expected={expected}"


@cocotb.test()
async def test_ra_data_valid_zero(dut):
    """data_valid=0: memory does not update; acc_out is still combinational."""
    await reset(dut)

    # Write something with first_tile=1 first.
    await feed(dut, row=0, first_tile=1, psum=[10, 20, 30, 40])

    # Now drive data_valid=0. acc_out is still combinational from inputs, but mem stays.
    dut.data_valid.value = 0
    dut.row_idx.value = 0
    dut.first_tile.value = 0
    dut.psum_in.value = pack([99, 99, 99, 99])
    await Timer(SETTLE_NS, units="ns")
    # acc_out should be mem[0] + 99 = [109, 119, 129, 139].
    got = unpack(int(dut.acc_out.value))
    assert got == [109, 119, 129, 139], f"combinational acc_out: {got}"
    await RisingEdge(dut.clk)

    # Now read back row 0 with data_valid=1, first_tile=0, psum=0 → should equal stored value.
    got = await feed(dut, row=0, first_tile=0, psum=[0, 0, 0, 0])
    assert got == [10, 20, 30, 40], f"persistent value: {got}"


@cocotb.test()
async def test_ra_back_to_back_kicks(dut):
    """First tile of a new kick overwrites stale memory."""
    await reset(dut)

    # Kick 1: tile 0 then tile 1.
    M = 2
    k1_t0 = [[100, 200, 300, 400], [50, 60, 70, 80]]
    k1_t1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    for row in range(M):
        await feed(dut, row=row, first_tile=1, psum=k1_t0[row])
    for row in range(M):
        await feed(dut, row=row, first_tile=0, psum=k1_t1[row])

    # Kick 2 starts with first_tile=1. Memory should overwrite.
    k2_t0 = [[7, 8, 9, 10], [11, 12, 13, 14]]
    for row in range(M):
        got = await feed(dut, row=row, first_tile=1, psum=k2_t0[row])
        assert got == k2_t0[row], f"kick2 row={row}: got={got} expected={k2_t0[row]}"
