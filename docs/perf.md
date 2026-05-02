# tinyNPU performance notes

Cycle counts measured at the APB level: from the cycle when `CTRL[0]=1`
is written until `STATUS.BUSY` returns to 0. All measurements use the
all-ones synthetic workload (`A=1`, `W=1`), no bias / ReLU / requantize,
single output `M`.

The measurements live in:

- `tb/test_top/test_top.py::test_top_4x4_cycle_count`
- `tb/test_top/test_top.py::test_top_4x4_cycle_count_n8k8`
- `tb/test_top_8x8/test_top_8x8.py::test_top_8x8_cycle_count`

## Single-tile latency

For a problem that fits in a single `(N tile, K tile)` slot the engine
sees no tile-loop overhead — it does one `LOAD_W → COMPUTE → WRITEBACK`
sequence.

| Array | M | N | K | MACs | Cycles | MACs / cycle | PE utilisation |
|-------|---|---|---|------|--------|--------------|----------------|
| 4×4   | 4 | 4 | 4 | 64   | 20     | 3.20         | 20.0 %         |
| 8×8   | 4 | 8 | 8 | 256  | 28     | 9.14         | 14.3 %         |

Pipeline latency from `if_start` (entry of `S_COMPUTE`) to the first
valid output beat is `ROWS + COLS + 1` cycles
(`= 1` FSM startup `+ 1` SRAM read `+ ROWS` PE column depth
`+ (COLS-1)` deepest unskew lane). For M=4 the steady-state output
window is M=4 cycles, and the writer adds 1 cycle for `ow_done`.
That accounts for `~ ROWS + COLS + M + 1` cycles inside `S_COMPUTE`
plus `≈ 6` cycles of FSM/SRAM/loader overhead — matching the 20 vs
28 numbers above.

PE utilisation is dominated by the fill/drain pipeline tax: for M=4 the
useful compute window (M=4 valid beats × `ROWS*COLS` PEs) is small
relative to the latency. The next subsection shows how this changes
when the problem walks across multiple tiles.

## Same problem, different array sizes

A direct apples-to-apples comparison: the M=4, N=8, K=8 GEMM is
exactly one tile on the 8×8 array but takes four tiles
(2 K tiles × 2 N tiles) on the 4×4 array.

| Array | Tiles | Cycles | MACs | MACs / cycle | Speedup vs 4×4 |
|-------|-------|--------|------|--------------|----------------|
| 4×4   | 4     | 72     | 256  | 3.56         | 1.00×          |
| 8×8   | 1     | 28     | 256  | 9.14         | 2.57×          |

The 8×8 array uses 4× the PEs; the 2.57× speedup on this small
problem (M=4) is below the ideal 4× because the pipeline fill/drain
tax grows as `ROWS + COLS`. With deeper M the utilisation improves on
both arrays — the per-tile fill cost is amortised over more output
rows.

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
array, corrupting the output. The fix is in `rtl/ctrl_fsm.sv`:
the COMPUTE termination now waits for `compute_done`
(driven by `data_done_pulse` in `tinyNPU_top`) instead of
`if_done`. ifm_feeder still emits its own done pulse, but the top
module ties it into the unused-signal collector — the data-side drain
is what gates progress.

## Future work

These numbers don't include cocotb coverage. Phase 14 plans to wire
`verilator --coverage` and cocotb's line/toggle coverage counters into
`scripts/run_all.sh`. Larger M (e.g., 16, 32) is the most natural way
to reduce the pipeline tax and push PE utilisation toward 90+%.
