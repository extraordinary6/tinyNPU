# Shared cocotb configuration for all tinyNPU testbenches.
# Each test Makefile does:
#   include ../common/cocotb.mk
#   VERILOG_SOURCES := ...
#   TOPLEVEL        := ...
#   MODULE          := ...
#   include $(COCOTB_MK)/Makefile.sim
#
# Configuration via environment variables (override in shell or Makefile.local):
#   PYTHON_BIN          MSYS2-style path (e.g. /d/anaconda/envs/py37/python.exe)
#                       to a Python 3.7 interpreter that has cocotb 1.8.1 installed.
#   PYTHONHOME_WINPATH  Windows-style path (e.g. D:/anaconda/envs/py37) of the
#                       same env's prefix. vvp.exe is a native Windows binary
#                       and reads PYTHONHOME from the process environment, but
#                       cocotb's Makefile.inc sets it to an MSYS2 path that vvp
#                       cannot understand. SIM_CMD_PREFIX overrides it.
#
# See README.md "Toolchain & setup" for installation details.

PYTHON_BIN          ?= /d/anaconda/envs/py37/python.exe
PYTHONHOME_WINPATH  ?= D:/anaconda/envs/py37

export PYTHON_BIN

COCOTB_MK := $(shell $(PYTHON_BIN) -m cocotb.config --share)/makefiles

export MAKE := $(shell which make)

# Force vvp's PYTHONHOME to a Windows-style path at invocation time.
# Without this, vvp can't find Python's stdlib and dies with a fatal init error.
SIM_CMD_PREFIX := PYTHONHOME=$(PYTHONHOME_WINPATH)

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus
