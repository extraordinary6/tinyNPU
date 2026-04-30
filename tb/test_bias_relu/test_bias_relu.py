"""Cocotb tests for rtl/bias_relu.sv (combinational bias + ReLU)."""

from __future__ import annotations

import random

import cocotb
from cocotb.triggers import Timer


LANES = 4
P_W = 32
SETTLE_NS = 1


def s32(x: int) -> int:
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x & (1 << 31) else x


def pack(vals):
    val = 0
    for c, v in enumerate(vals):
        val |= (int(v) & 0xFFFF_FFFF) << (c * P_W)
    return val


def unpack(raw: int):
    return [s32((raw >> (c * P_W)) & 0xFFFF_FFFF) for c in range(LANES)]


def model(acc, bias, bias_en, relu_en):
    out = []
    for c in range(LANES):
        s = acc[c] + bias[c] if bias_en else acc[c]
        if relu_en and s < 0:
            s = 0
        out.append(s)
    return out


async def apply_and_check(dut, *, acc, bias, bias_en, relu_en):
    dut.acc_in.value = pack(acc)
    dut.bias_in.value = pack(bias)
    dut.bias_en.value = bias_en
    dut.relu_en.value = relu_en
    await Timer(SETTLE_NS, units="ns")
    got = unpack(int(dut.data_out.value))
    expected = model(acc, bias, bias_en, relu_en)
    assert got == expected, (
        f"acc={acc} bias={bias} bias_en={bias_en} relu_en={relu_en}\n"
        f"got={got} expected={expected}"
    )


@cocotb.test()
async def test_brelu_bypass(dut):
    """bias_en=0 relu_en=0: data_out = acc_in."""
    await apply_and_check(dut, acc=[5, -7, 0, 100], bias=[1, 1, 1, 1],
                          bias_en=0, relu_en=0)
    await apply_and_check(dut, acc=[-1, -2, -3, -4], bias=[100, 100, 100, 100],
                          bias_en=0, relu_en=0)


@cocotb.test()
async def test_brelu_bias_only(dut):
    """bias_en=1 relu_en=0: bias is added, negatives pass through."""
    await apply_and_check(dut, acc=[10, -10, 100, -100], bias=[1, 2, -3, -4],
                          bias_en=1, relu_en=0)


@cocotb.test()
async def test_brelu_relu_only(dut):
    """relu_en=1 bias_en=0: clip negatives to 0, keep zero and positives."""
    await apply_and_check(dut, acc=[-1, 0, 1, -2_000_000_000],
                          bias=[0, 0, 0, 0], bias_en=0, relu_en=1)


@cocotb.test()
async def test_brelu_both(dut):
    """Both enabled: ReLU(acc + bias). Verify the boundary at 0 (kept)."""
    await apply_and_check(dut, acc=[5, -5, 100, -100], bias=[-5, 5, -50, 50],
                          bias_en=1, relu_en=1)
    await apply_and_check(dut, acc=[0, 1, -1, 0], bias=[0, -1, 1, 0],
                          bias_en=1, relu_en=1)


@cocotb.test()
async def test_brelu_random(dut):
    """Random fuzz across all 4 (bias_en, relu_en) combinations."""
    rng = random.Random(0xB1A5)
    for _ in range(200):
        acc = [rng.randint(-(1 << 30), (1 << 30)) for _ in range(LANES)]
        bias = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(LANES)]
        be = rng.randint(0, 1)
        re = rng.randint(0, 1)
        await apply_and_check(dut, acc=acc, bias=bias, bias_en=be, relu_en=re)
