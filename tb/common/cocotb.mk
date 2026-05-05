# Shared cocotb configuration for all tinyNPU testbenches.
# Each test Makefile does:
#   include ../common/cocotb.mk
#   VERILOG_SOURCES := ...
#   TOPLEVEL        := ...
#   MODULE          := ...
#   include $(COCOTB_MK)/Makefile.sim
#
# Configuration via environment variables (override in shell or Makefile.local):
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
SIM           ?= icarus
