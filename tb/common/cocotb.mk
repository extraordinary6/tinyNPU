# Shared cocotb configuration for all tinyNPU testbenches.
# Each test Makefile does:
#   include ../common/cocotb.mk
#   VERILOG_SOURCES := ...
#   TOPLEVEL        := ...
#   MODULE          := ...
#   include $(COCOTB_MK)/Makefile.sim
#
# Configuration via environment variables (override in shell or Makefile.local):
#   SIM                 cocotb simulator backend. Default: icarus. Pass
#                       SIM=verilator (or run `make coverage`) to use the
#                       Verilator path. Verilator is required for line/toggle
#                       coverage collection; Icarus has no equivalent.
#   PYTHON_BIN          path to a Python interpreter with cocotb 1.8.1 installed.
#                       Default: python3 on Linux, /d/anaconda/.../python.exe on MSYS2.
#   PYTHONHOME_WINPATH  (MSYS2 only) Windows-style path of the same env's prefix.
#                       vvp.exe is a native Windows binary and reads PYTHONHOME from
#                       the process environment; cocotb's Makefile.inc sets it to an
#                       MSYS2 path that vvp cannot understand. SIM_CMD_PREFIX overrides it.

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
PYTHON_BIN          ?= python3
PYTHONHOME_WINPATH  ?=
else
PYTHON_BIN          ?= /d/anaconda/envs/py37/python.exe
PYTHONHOME_WINPATH  ?= D:/anaconda/envs/py37
endif

export PYTHON_BIN

COCOTB_MK := $(shell $(PYTHON_BIN) -m cocotb.config --share)/makefiles

export MAKE := $(shell which make)

# On MSYS2, force vvp's PYTHONHOME to a Windows-style path at invocation time.
# On Linux this is a no-op (PYTHONHOME_WINPATH is empty, SIM_CMD_PREFIX unset).
ifneq ($(PYTHONHOME_WINPATH),)
SIM_CMD_PREFIX := PYTHONHOME=$(PYTHONHOME_WINPATH)
endif

TOPLEVEL_LANG ?= verilog

# When `make coverage` is the goal, force SIM=verilator at the outer
# make's parse time. Cocotb's Makefile.icarus does a parse-time
# `$(error)` if iverilog isn't installed, so the outer make would die
# before the coverage recipe ever runs on a Verilator-only CI runner.
ifneq (,$(filter coverage,$(MAKECMDGOALS)))
    SIM := verilator
endif
SIM ?= icarus

# Pin the default goal to cocotb's `sim` target. Without this, the `coverage`
# rule below (the first explicit rule encountered when this fragment is
# included) would otherwise steal the default-goal slot from cocotb's
# Makefile.sim, breaking `make` invocations that expect to run the regression.
.DEFAULT_GOAL := sim

# `make coverage` re-runs the testbench under Verilator with line+toggle
# coverage enabled, then converts the resulting coverage.dat into lcov
# format (coverage.info, alongside the testbench's Makefile). Verilator is
# required because Icarus emits no coverage data. SIM is passed as a CLI
# override to the recursive make so it propagates reliably to the sub-make
# regardless of how env-var inheritance behaves on the host.
.PHONY: coverage
coverage:
	$(MAKE) SIM=verilator EXTRA_ARGS="--coverage-line --coverage-toggle"
	@if [ ! -f coverage.dat ]; then \
		echo "coverage: no coverage.dat produced (verilator coverage build failed?)" >&2; \
		exit 1; \
	fi
	verilator_coverage --write-info coverage.info coverage.dat
	@echo ">>> wrote $$(pwd)/coverage.info"

# Cocotb's per-simulator Makefile defines `clean::` (double-colon) for
# build/dump artefacts; append our coverage outputs so `make clean` is
# self-contained.
clean::
	$(RM) coverage.dat coverage.info
