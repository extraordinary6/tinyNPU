# tinyNPU

A small, hackable NPU written in SystemVerilog and verified with cocotb +
numpy golden models. INT8 × INT8 → INT32 GEMM core, 4×4 weight-stationary
systolic array, APB3 register interface.

Designed as a teaching / experimentation vehicle: every module has its own
testbench, the v1 datapath end-to-end fits in ~300 lines of RTL, and the
roadmap (`plan.md`) is split into bite-sized phases.

For full architectural detail and per-phase milestones, see [`plan.md`](./plan.md).

## Status

Phase 0 through 7 are complete. The full v1 datapath runs GEMM, optional ReLU,
and TFLite-lite requantize, all driven through the APB3 CSR interface.

| Phase | Content | State |
|-------|---------|-------|
| 0 | Toolchain skeleton (DFF smoke test) | done |
| 1 | `pe.sv` — single MAC | done |
| 2 | `systolic_array.sv` — 4×4 PE array | done |
| 3 | `accumulator` / `bias_relu` / `requantize` | done |
| 4 | `sram_wrapper` / `ifm_feeder` / `weight_loader` / `ofm_writer` | done |
| 5 | `apb_csr` (APB3 slave) + `ctrl_fsm` (top FSM) | done |
| 6 | `tinyNPU_top` end-to-end integration | done |
| 7 | Verilator strict lint clean, basic regression script | done |
| 8+ | Per-channel requantize / bias loader / K-tile / N-tile / 8×8 / Conv2D / coverage | planned (`plan.md` §7+) |

### v1 capabilities (after phase 7)

- Single 4×4 GEMM tile per kick: `C = A @ W`, A is `M × 4`, W is `4 × 4`,
  output is `M × 4` INT8 (or low byte of INT32 in bypass mode).
- `FLAGS.RELU_EN`: ReLU on the accumulator output.
- `FLAGS.REQ_EN`: TFLite-lite requantize (signed mult-shift-saturate).
- M / N / K = 0 → `STATUS.ERR` pulses for one cycle.
- Verilator `--lint-only -Wall` passes with zero warnings.

### Known v1 limitations (addressed by phases 8+)

- K must equal 4 (no K-tile accumulation across multiple weight tiles).
- N must equal 4 (no N-tile sweep).
- `bias_in` is hard-wired to 0; `FLAGS.BIAS_EN` is currently a no-op.
- `req_mult` / `req_shift` are global, not per-channel.

Pipeline latency from `if_start` to first OFM write = 9 cycles
(FSM startup + SRAM read + 4-deep PE column + 3-deep unskew).

## Test summary

13 cocotb testbenches, 64 cases total. Run them all with `bash scripts/run_all.sh`.

| Testbench | Cases | DUT |
|-----------|-------|-----|
| `tb/test_dff/`             | 2 | toolchain smoke |
| `tb/test_pe/`              | 6 | `rtl/pe.sv` |
| `tb/test_systolic_array/`  | 5 | `rtl/systolic_array.sv` |
| `tb/test_accumulator/`     | 5 | `rtl/accumulator.sv` |
| `tb/test_bias_relu/`       | 5 | `rtl/bias_relu.sv` |
| `tb/test_requantize/`      | 7 | `rtl/requantize.sv` |
| `tb/test_sram_wrapper/`    | 4 | `rtl/sram_wrapper.sv` |
| `tb/test_weight_loader/`   | 4 | `rtl/weight_loader.sv` (+sram harness) |
| `tb/test_ofm_writer/`      | 4 | `rtl/ofm_writer.sv` (+sram harness) |
| `tb/test_ifm_feeder/`      | 5 | `rtl/ifm_feeder.sv` (+sram harness) |
| `tb/test_apb_csr/`         | 7 | `rtl/apb_csr.sv` |
| `tb/test_ctrl_fsm/`        | 5 | `rtl/ctrl_fsm.sv` |
| `tb/test_top/`             | 5 | `rtl/tinyNPU_top.sv` (end-to-end via APB) |

## Toolchain & setup

The project has been developed and verified on **Windows + MSYS2 + Anaconda Python 3.7
+ Icarus Verilog**. Linux should also work with minor adjustments, but is not
tested here.

### Required components

| Tool | Tested version | How to install |
|------|----------------|----------------|
| MSYS2 (`bash` + `make`) | 2024+ | https://www.msys2.org/, then `pacman -S make` |
| Icarus Verilog | 12.0+ (devel) | https://bleyer.org/icarus/ (Windows builds) |
| Python | 3.7.x | conda recommended: `conda create -n py37 python=3.7` |
| `cocotb` | **1.8.1** (last version supporting Py3.7) | `pip install cocotb==1.8.1` |
| `numpy` | <1.22 | pinned in `requirements.txt` |
| `pytest` | <7.5 | pinned in `requirements.txt` |
| Verilator (lint only) | 5.026 | `pacman -S mingw-w64-x86_64-verilator` |

### Quick install

```bash
# 1. install Python deps into your Python 3.7 env
<your-py37-env>/python.exe -m pip install -r requirements.txt

# 2. ensure iverilog is on PATH (or note its absolute path; cocotb finds it via PATH)
iverilog -V

# 3. (optional) install Verilator for strict linting
pacman -S mingw-w64-x86_64-verilator
```

### Per-machine configuration

Two paths in `tb/common/cocotb.mk` need to point at your Python 3.7 install.
They have sensible defaults; override either via environment variable or by
editing the file.

```make
PYTHON_BIN          ?= /d/anaconda/envs/py37/python.exe   # MSYS2-style path
PYTHONHOME_WINPATH  ?= D:/anaconda/envs/py37              # Windows-style path of the same env
```

Why two: `make`'s `$(shell ...)` runs under MSYS2 and needs the `/d/...` form;
`vvp.exe` is a native Windows binary and reads `PYTHONHOME` from the process
environment in `D:/...` form. We set the latter via `SIM_CMD_PREFIX`.

The Verilator wrapper script needs Perl's `Pod::Usage`, which the MSYS2 system
Perl is missing. `scripts/lint.sh` calls `verilator_bin.exe` directly to skip
the wrapper, and exports `VERILATOR_ROOT` because the binary's compiled-in
default path doesn't survive on Windows. Both can be overridden:

```bash
VERILATOR_BIN=/path/to/verilator_bin.exe \
VERILATOR_ROOT=/path/to/share/verilator \
bash scripts/lint.sh
```

### One known cocotb 1.8.1 patch

`cocotb/share/makefiles/Makefile.inc` (inside the installed cocotb) has one
line that calls `cocotb-config --python-bin`, which on Windows returns a
backslash path that MSYS2 `make` mangles. Symptom: 4 lines of
`Danacondaenvspy37python.exe: command not found` noise during every run.
Fix once, after every cocotb reinstall:

```diff
- IS_VENV=$(shell $(shell cocotb-config --python-bin) -c '...')
+ IS_VENV=$(shell $(PYTHON_BIN) -c '...')
```

It's harmless if you skip the patch — just noisy.

## Running the tests

```bash
# single testbench
cd tb/test_top && make

# all testbenches (clean + run, returns non-zero on failure)
bash scripts/run_all.sh

# strict lint (Verilator)
bash scripts/lint.sh
```

Each testbench drops `sim_build/` and `results.xml` next to the Makefile; both
are gitignored.

## Repository layout

```
tinyNPU/
├── plan.md                  Detailed roadmap, architecture, register map, coding rules
├── README.md                This file
├── requirements.txt         cocotb==1.8.1, numpy<1.22, pytest<7.5
├── rtl/                     SystemVerilog sources (12 modules)
│   ├── pe.sv
│   ├── systolic_array.sv
│   ├── accumulator.sv
│   ├── bias_relu.sv
│   ├── requantize.sv
│   ├── sram_wrapper.sv
│   ├── ifm_feeder.sv
│   ├── weight_loader.sv
│   ├── ofm_writer.sv
│   ├── apb_csr.sv
│   ├── ctrl_fsm.sv
│   └── tinyNPU_top.sv
├── tb/
│   ├── common/              cocotb.mk shared config, golden_model.py (numpy reference)
│   └── test_*/              one cocotb testbench per RTL module + end-to-end test_top
└── scripts/
    ├── run_all.sh           batch regression
    └── lint.sh              Verilator strict lint
```

## Coding conventions

The strict rules live in `plan.md` §8. Highlights:

- One `always_ff` drives one signal (unless reset / branch shape is identical).
- Only `logic`; no `reg` / `wire`.
- Synchronous active-low reset `rst_n`.
- All literals carry width and base (`32'sd0`, `8'sh7F`).
- Instantiations use named connections only.
- No `initial` / `#delay` / `$display` / `force` in synthesisable RTL.

## Roadmap beyond v1

Phases 8 through 14 are described in detail in `plan.md`. In short:

| Phase | Headline goal | Effort |
|-------|---------------|--------|
| 8  | Per-channel requantize | 1–2 d |
| 9  | Bias loader (FLAGS.BIAS_EN actually works) | 1 d |
| 10 | K-tile accumulation (K > 4) | 2 d |
| 11 | N-tile sweep (N > 4) + parameterized unskew | 1–2 d |
| 12 | 8×8 PE array | 1–2 d |
| 13 | Conv2D via Python im2col driver | 1 d |
| 14 | Coverage + cycle-count benchmarking + CI | 1–2 d |

## License

Released under the MIT License. See [`LICENSE`](./LICENSE) for the full text.

