"""Cocotb tests for rtl/apb_csr.sv (APB3 slave with CSR file).

APB3 transfer:
  cycle T: SETUP   psel=1 penable=0
  cycle T+1: ACCESS psel=1 penable=1 (slave drives prdata, pready, pslverr)
  cycle T+2: back to IDLE psel=0
"""

from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


CLK_NS = 10
SETTLE_NS = 1

ID_MAGIC = 0x4E50_5500

# Address map (must match RTL).
A_ID = 0x000
A_CTRL = 0x004
A_STATUS = 0x008
A_M = 0x00C
A_N = 0x010
A_K = 0x014
A_IFM = 0x018
A_W = 0x01C
A_OFM = 0x020
A_BIAS = 0x024
A_FLAGS = 0x028
A_REQ_MULT = 0x02C
A_REQ_SHIFT = 0x030
A_REQ_MULT_BASE = 0x034
A_REQ_SHIFT_BASE = 0x038

RW_REGS = [
    ("M", A_M, "m_count"),
    ("N", A_N, "n_count"),
    ("K", A_K, "k_count"),
    ("IFM", A_IFM, "ifm_base"),
    ("W", A_W, "w_base"),
    ("OFM", A_OFM, "ofm_base"),
    ("BIAS", A_BIAS, "bias_base"),
    ("FLAGS", A_FLAGS, "flags"),
    ("REQ_MULT", A_REQ_MULT, "req_mult"),
    ("REQ_SHIFT", A_REQ_SHIFT, "req_shift"),
    ("REQ_MULT_BASE", A_REQ_MULT_BASE, "req_mult_base"),
    ("REQ_SHIFT_BASE", A_REQ_SHIFT_BASE, "req_shift_base"),
]


async def reset(dut, cycles=2):
    dut.presetn.value = 0
    dut.psel.value = 0
    dut.penable.value = 0
    dut.pwrite.value = 0
    dut.paddr.value = 0
    dut.pwdata.value = 0
    dut.busy.value = 0
    dut.done.value = 0
    dut.err.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.pclk)
    dut.presetn.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")


async def apb_write(dut, addr, data):
    """One APB3 write transfer; returns (pslverr) observed on the access cycle."""
    # SETUP
    dut.psel.value = 1
    dut.penable.value = 0
    dut.pwrite.value = 1
    dut.paddr.value = addr
    dut.pwdata.value = data & 0xFFFF_FFFF
    await RisingEdge(dut.pclk)
    # ACCESS
    dut.penable.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")
    err = int(dut.pslverr.value)
    rdy = int(dut.pready.value)
    # IDLE
    dut.psel.value = 0
    dut.penable.value = 0
    dut.pwrite.value = 0
    return rdy, err


async def apb_read(dut, addr):
    dut.psel.value = 1
    dut.penable.value = 0
    dut.pwrite.value = 0
    dut.paddr.value = addr
    await RisingEdge(dut.pclk)
    dut.penable.value = 1
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")
    data = int(dut.prdata.value)
    rdy = int(dut.pready.value)
    err = int(dut.pslverr.value)
    dut.psel.value = 0
    dut.penable.value = 0
    return data, rdy, err


@cocotb.test()
async def test_csr_reset_defaults(dut):
    """All RW registers reset to 0; ID is the magic constant."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    data, rdy, err = await apb_read(dut, A_ID)
    assert rdy == 1 and err == 0
    assert data == ID_MAGIC, f"ID=0x{data:08x}"

    for name, addr, _ in RW_REGS:
        data, _, err = await apb_read(dut, addr)
        assert err == 0
        assert data == 0, f"{name} not zero after reset: 0x{data:08x}"


@cocotb.test()
async def test_csr_rw_roundtrip(dut):
    """Write each RW register, read it back."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    payload = {
        A_M: 0x0000_0010, A_N: 0x0000_0020, A_K: 0x0000_0040,
        A_IFM: 0x1000, A_W: 0x2000, A_OFM: 0x3000, A_BIAS: 0x4000,
        A_FLAGS: 0x0000_000F, A_REQ_MULT: 0xFFFF_F000, A_REQ_SHIFT: 0x0000_001F,
        A_REQ_MULT_BASE: 0x5000, A_REQ_SHIFT_BASE: 0x6000,
    }
    for addr, data in payload.items():
        rdy, err = await apb_write(dut, addr, data)
        assert rdy == 1 and err == 0
    for addr, expected in payload.items():
        got, _, err = await apb_read(dut, addr)
        assert err == 0 and got == expected, f"addr=0x{addr:03x} got=0x{got:08x} expected=0x{expected:08x}"


@cocotb.test()
async def test_csr_id_readonly(dut):
    """Writing to A_ID is silently ignored."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    rdy, err = await apb_write(dut, A_ID, 0xDEADBEEF)
    assert rdy == 1 and err == 0  # the address is mapped, but RO
    data, _, _ = await apb_read(dut, A_ID)
    assert data == ID_MAGIC


@cocotb.test()
async def test_csr_unmapped_pslverr(dut):
    """Unmapped address asserts pslverr in the ACCESS cycle."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    bad = 0x100
    _, rdy, err = await apb_read(dut, bad)
    assert rdy == 1 and err == 1, f"unmapped read: rdy={rdy} err={err}"
    rdy, err = await apb_write(dut, bad, 0x1234)
    assert rdy == 1 and err == 1


@cocotb.test()
async def test_csr_status_reflects_inputs(dut):
    """STATUS [0]=BUSY [1]=DONE [2]=ERR comes directly from inputs."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    for busy, done, err in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
        dut.busy.value = busy
        dut.done.value = done
        dut.err.value = err
        await RisingEdge(dut.pclk)
        await Timer(SETTLE_NS, units="ns")
        data, _, _ = await apb_read(dut, A_STATUS)
        expected = (err << 2) | (done << 1) | busy
        assert (data & 0x7) == expected, f"busy={busy} done={done} err={err}: STATUS=0x{data:08x}"


@cocotb.test()
async def test_csr_start_w1s_pulse(dut):
    """Writing CTRL[0]=1 pulses start_pulse for exactly one cycle."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)

    # SETUP: drive CTRL write
    dut.psel.value = 1
    dut.penable.value = 0
    dut.pwrite.value = 1
    dut.paddr.value = A_CTRL
    dut.pwdata.value = 0x1
    await RisingEdge(dut.pclk)
    # ACCESS
    dut.penable.value = 1
    await RisingEdge(dut.pclk)
    # After the access edge, start_q should be high. Settle and check.
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.start_pulse.value) == 1, "start_pulse should be high right after ACCESS edge"

    # Drop the bus, check pulse self-clears one cycle later.
    dut.psel.value = 0
    dut.penable.value = 0
    dut.pwrite.value = 0
    await RisingEdge(dut.pclk)
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.start_pulse.value) == 0, "start_pulse should self-clear"


@cocotb.test()
async def test_csr_start_w0_no_pulse(dut):
    """Writing CTRL with bit0=0 does not pulse start."""
    cocotb.start_soon(Clock(dut.pclk, CLK_NS, units="ns").start())
    await reset(dut)
    rdy, err = await apb_write(dut, A_CTRL, 0xFFFF_FFFE)  # bit0=0
    assert err == 0
    # After ACCESS, start_pulse should be 0.
    await Timer(SETTLE_NS, units="ns")
    assert int(dut.start_pulse.value) == 0
