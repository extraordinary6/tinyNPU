"""Cocotb tests for rtl/requantize.sv (per-lane TFLite-lite mult-shift-saturate, INT32 -> INT8).

Reference: plan.md §3.1.1 / §阶段 8 and tb/common/golden_model.py::requantize.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import cocotb
from cocotb.triggers import Timer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "..", "common"))
from golden_model import (  # noqa: E402
    requantize as golden_requantize,
    requantize_per_channel as golden_requantize_pc,
)


LANES = 4
P_W = 32
O_W = 8
SETTLE_NS = 1


def s8(x: int) -> int:
    x &= 0xFF
    return x - 0x100 if x & 0x80 else x


def s32(x: int) -> int:
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x & (1 << 31) else x


def pack32(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0xFFFF_FFFF) << (c * P_W)
    return val


def pack_mults(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0xFFFF_FFFF) << (c * P_W)
    return val


def pack_shifts(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0x3F) << (c * 6)
    return val


def unpack8(raw: int):
    return [s8((raw >> (c * O_W)) & 0xFF) for c in range(LANES)]


async def apply_and_check(dut, *, acc, mults, shifts, req_en, expected):
    if not isinstance(mults, (list, tuple)):
        mults = [mults] * LANES
    if not isinstance(shifts, (list, tuple)):
        shifts = [shifts] * LANES
    dut.acc_in.value = pack32(acc)
    dut.mult.value = pack_mults(mults)
    dut.shift.value = pack_shifts(shifts)
    dut.req_en.value = req_en
    await Timer(SETTLE_NS, units="ns")
    got = unpack8(int(dut.data_out.value))
    assert got == expected, (
        f"acc={acc} mult={mults} shift={shifts} req_en={req_en}\n"
        f"got={got} expected={expected}"
    )


def model_requantize(acc, mult, shift):
    arr = np.array(acc, dtype=np.int32)
    return [int(x) for x in golden_requantize(arr, mult, shift)]


def model_requantize_pc(acc, mults, shifts):
    arr = np.array(acc, dtype=np.int32)
    return [int(x) for x in golden_requantize_pc(arr, mults, shifts)]


@cocotb.test()
async def test_req_bypass(dut):
    """req_en=0: data_out lane = acc[7:0] (low 8 bits, sign-extended)."""
    cases = [
        ([0, 1, -1, 256], [0, 1, -1, 0]),
        ([127, 128, -128, -129], [127, -128, -128, 127]),
        ([0x12345678, -0x12345678, 0xFF, -1], [0x78, -0x78, -1, -1]),
    ]
    for acc, exp in cases:
        await apply_and_check(dut, acc=acc, mults=1, shifts=0, req_en=0, expected=exp)


@cocotb.test()
async def test_req_shift_zero(dut):
    """shift=0: no rounding bias, product is taken as-is then saturated."""
    acc = [10, -10, 200, -200]
    expected = model_requantize(acc, 1, 0)
    await apply_and_check(dut, acc=acc, mults=1, shifts=0, req_en=1, expected=expected)
    acc2 = [127, -128, 1, -1]
    exp2 = model_requantize(acc2, 1000, 0)
    await apply_and_check(dut, acc=acc2, mults=1000, shifts=0, req_en=1, expected=exp2)


@cocotb.test()
async def test_req_shift_max(dut):
    """shift=31: very large divisor, most outputs are 0 or +/-1 with rounding."""
    acc = [1 << 30, -(1 << 30), (1 << 29), -(1 << 29)]
    expected = model_requantize(acc, 1, 31)
    await apply_and_check(dut, acc=acc, mults=1, shifts=31, req_en=1, expected=expected)


@cocotb.test()
async def test_req_round_half_up(dut):
    """Verify round-half-up: +0.5 -> +1, -0.5 -> 0 (not -1)."""
    acc = [1, -1, 3, -3]
    expected = model_requantize(acc, 1, 1)
    await apply_and_check(dut, acc=acc, mults=1, shifts=1, req_en=1, expected=expected)
    assert expected == [1, 0, 2, -1]


@cocotb.test()
async def test_req_saturate(dut):
    """Output saturates at +127 / -128."""
    acc = [10000, -10000, 1, -1]
    expected = model_requantize(acc, 1 << 20, 0)
    assert expected[0] == 127 and expected[1] == -128
    await apply_and_check(dut, acc=acc, mults=1 << 20, shifts=0, req_en=1, expected=expected)


@cocotb.test()
async def test_req_signed_mult(dut):
    """Negative mult flips sign of result."""
    acc = [10, -10, 100, -100]
    exp_pos = model_requantize(acc, 1, 0)
    exp_neg = model_requantize(acc, -1, 0)
    await apply_and_check(dut, acc=acc, mults=1, shifts=0, req_en=1, expected=exp_pos)
    await apply_and_check(dut, acc=acc, mults=-1, shifts=0, req_en=1, expected=exp_neg)


@cocotb.test()
async def test_req_random(dut):
    """Random fuzz across acc / mult / shift broadcast (golden = scalar requantize)."""
    rng = np.random.default_rng(0xDEADBEEF)
    for _ in range(150):
        acc = list(rng.integers(-(1 << 31), 1 << 31, size=LANES, dtype=np.int64))
        mult = int(rng.integers(-(1 << 20), 1 << 20))
        shift = int(rng.integers(0, 32))
        expected = model_requantize(acc, mult, shift)
        await apply_and_check(dut, acc=acc, mults=mult, shifts=shift, req_en=1, expected=expected)


@cocotb.test()
async def test_req_per_channel(dut):
    """Each lane uses an independent (mult, shift)."""
    acc = [1000, -1000, 250, -250]
    mults = [1 << 20, 1 << 18, -(1 << 20), 1 << 22]
    shifts = [12, 8, 14, 18]
    expected = model_requantize_pc(acc, mults, shifts)
    await apply_and_check(dut, acc=acc, mults=mults, shifts=shifts, req_en=1, expected=expected)


@cocotb.test()
async def test_req_per_channel_random(dut):
    """Random fuzz with per-lane mult/shift, golden via requantize_per_channel."""
    rng = np.random.default_rng(0xC0FFEE)
    for _ in range(150):
        acc = list(rng.integers(-(1 << 31), 1 << 31, size=LANES, dtype=np.int64))
        mults = [int(rng.integers(-(1 << 20), 1 << 20)) for _ in range(LANES)]
        shifts = [int(rng.integers(0, 32)) for _ in range(LANES)]
        expected = model_requantize_pc(acc, mults, shifts)
        await apply_and_check(dut, acc=acc, mults=mults, shifts=shifts, req_en=1, expected=expected)
