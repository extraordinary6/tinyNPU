# tinyNPU performance notes

Cycle counts measured at the APB level: from the cycle when `CTRL[0]=1`
is written until `STATUS.BUSY` returns to 0. All measurements use the
all-ones synthetic workload (`A = 1`, `W = 1`), no bias / ReLU / requantize.
"PE utilisation" is `MACs / (cycles × ROWS × COLS)`.

The captured measurements live in two cocotb sweeps and are part of the
regression — `bash scripts/run_all.sh` re-prints every `PERF` line:

- `tb/test_top/test_top.py`
  - `test_top_4x4_cycle_count` (4×4, M=4, N=K=4)
  - `test_top_4x4_cycle_count_n8k8` (4×4, M=4, N=K=8)
  - `test_top_4x4_perf_singletile_m{8,16}`
  - `test_top_4x4_perf_n8k8_m{8,16}`
- `tb/test_top_8x8/test_top_8x8.py`
  - `test_top_8x8_cycle_count` (8×8, M=4, N=K=8)
  - `test_top_8x8_perf_singletile_m{8,16}`
  - `test_top_8x8_perf_n16k16_m{8,16}`

## Theoretical lower bound

For a single `(N tile, K tile)` slot the engine runs one
`LOAD_W → COMPUTE → WRITEBACK` sequence. The compute window is bounded
by the systolic pipeline:

```
cycles_per_tile ≈ M                     // valid output beats
                + (ROWS - 1)            // stagger fill (top row first)
                + (COLS - 1)            // rightmost column unskew
                + 1                     // FSM IDLE → LOAD_W startup
                + 1                     // SRAM read latency
                + ~6                    // LOAD_W + WRITEBACK + drain
              ≈ M + (ROWS + COLS) + 8
```

Equivalently: **`cycles ≈ M + 16`** for the 4×4 array, **`cycles ≈ M + 24`**
for the 8×8 array. The measurements below match those formulae to the
cycle.

## Single-tile sweep over M

```
4×4 array (16 PEs)              8×8 array (64 PEs)
──────────────────────────      ───────────────────────────
M    cycles  MAC/cy  util       M    cycles  MAC/cy  util
4    20      3.20    20.0 %     4    28       9.14   14.3 %
8    24      5.33    33.3 %     8    32      16.00   25.0 %
16   32      8.00    50.0 %     16   40      25.60   40.0 %
```

`cycles - M` is constant per array — exactly the fill/drain pipeline tax
predicted by the formula. All useful compute happens in the M valid
beats; everything else is overhead. Doubling M roughly doubles MAC/cycle
while overhead stays put, which is why utilisation rises sharply with M.

The 8×8 array ships 4× the PEs but pays a larger pipeline tax (24 vs 16
overhead cycles). To break even on utilisation against 4×4 single-tile
M=4 (20 %), 8×8 needs M≈8 (25 %); to beat it convincingly, M≈16 (40 %).

## Multi-tile sweep over M

The engine reuses LOAD_W/COMPUTE across `K_TILES × N_TILES` tiles per
kick; WRITEBACK fires only on the last K tile of each N tile. Per-tile
cost is therefore close to single-tile but slightly smaller (WRITEBACK
amortises across K tiles within an N tile):

```
4×4 array, N=8, K=8 (2 N tiles × 2 K tiles = 4 tiles)
M    cycles  cycles/tile  MAC/cy  util
4    72      18.0         3.56    22.2 %
8    88      22.0         5.82    36.4 %
16   120     30.0         8.53    53.3 %

8×8 array, N=16, K=16 (2 N tiles × 2 K tiles = 4 tiles)
M    cycles  cycles/tile  MAC/cy   util
8    120     30.0         17.07   26.7 %
16   152     38.0         26.95   42.1 %
```

`cycles_per_tile = M + 14` on 4×4 vs single-tile `M + 16`; the 2-cycle
saving comes from skipping WRITEBACK on the first K tile of each N tile
(`K_TILES = 2` halves the writeback count vs running them as four
independent kicks). The same 2-cycle gap holds on the 8×8 numbers
(`M + 22` vs `M + 24`).

## Same problem, different array sizes

A direct apples-to-apples view: M=4, N=8, K=8 is exactly one tile on the
8×8 array but lowers to four tiles (2 K tiles × 2 N tiles) on 4×4.

| Array | Tiles | Cycles | MACs | MAC/cycle | Speedup vs 4×4 |
|-------|-------|--------|------|-----------|----------------|
| 4×4   | 4     | 72     | 256  | 3.56      | 1.00×          |
| 8×8   | 1     | 28     | 256  | 9.14      | 2.57×          |

The 8×8 array uses 4× the PEs; the 2.57× speedup on this small problem
is below the ideal 4× because the pipeline fill/drain tax grows with
`ROWS + COLS`. With deeper M the utilisation gap shrinks — see the 4×4
M=16 single-tile run (50.0 %) and the 8×8 M=16 single-tile run (40.0 %).

## Hotspots

The fill/drain pipeline tax is the only meaningful hotspot at this
problem size; every other phase (LOAD_W, LOAD_BIAS, LOAD_REQ, WRITEBACK)
is at most a handful of cycles. From the formulae:

- For `M = ROWS + COLS`, the COMPUTE window is half pipeline tax, half
  useful compute → utilisation ≈ 50 %.
- Asymptote: `cycles → M` (no tax) → utilisation → 100 % as M grows.

In practice the easiest knob is **M**. Doubling M ≈ doubles useful work
while overhead stays put; for an 8×8 array, M=64 single-tile would land
at `cycles ≈ 88`, `MACs = 4096`, **utilisation ≈ 73 %**. Larger M is
free RTL-side — `M_W = 16` and `M_MAX = 64` parameters in
`tinyNPU_top.sv` already cover it.

The only cycle previously discovered to be load-bearing was the
`if_done` → `compute_done` swap in phase 12 (see below).

## Notes on the 8×8 → ctrl_fsm interaction

Phase 12 surfaced a timing rule that the 4×4 design hid. ctrl_fsm
originally left `S_COMPUTE` on `if_done` (ifm_feeder's input-stream
done pulse). For ROWS=COLS=4, M=4, the gap from `if_done` to
`data_done_pulse` (last output beat through the unskew) is short
enough that the next K tile's `LOAD_W → w_load` lands one cycle after
the last valid beat — safe.

For ROWS=COLS=8, M=4, the gap stretches to `ROWS + COLS - M = 12`
cycles. Using `if_done` would let `w_load` overwrite PE weights while
the previous tile's last beats are still propagating through the
array, corrupting the output. The fix is in `rtl/ctrl_fsm.sv`: the
COMPUTE termination now waits for `compute_done` (driven by
`data_done_pulse` in `tinyNPU_top`) instead of `if_done`. ifm_feeder
still emits its own done pulse, but the top module ties it into the
unused-signal collector — the data-side drain is what gates progress.

## Continuous integration

Phase 14 wires the regression and lint scripts into
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Every push
and pull request runs `scripts/run_all.sh` (every cocotb testbench)
followed by `scripts/lint.sh` (Verilator strict lint) on an Ubuntu
runner with Python 3.7 + cocotb 1.8.1 + numpy<1.22 + Icarus 12 +
Verilator. Failures gate the merge.

## Future work — coverage

Cocotb 1.8.1 surfaces line/toggle counters when running under
`SIM=verilator` rather than Icarus. Switching the regression simulator
is a Makefile-level change but breaks Icarus's MSYS2 workarounds. A
small follow-up could:

1. Add a `SIM ?= icarus` Makefile knob that forwards to cocotb's own
   `SIM` selection.
2. Add a `make coverage` target in the testbenches that re-runs under
   Verilator with `-CFLAGS --coverage`.
3. Aggregate `coverage.info` artifacts in CI.

Out of scope for v1; the cycle-count sweeps above already cover the
end-to-end performance question that phase 14 set out to answer.
