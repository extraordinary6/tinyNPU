include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/bias_loader.sv) \
                   $(abspath bl_harness.sv)
TOPLEVEL        := bl_harness
MODULE          := test_bias_loader

include $(COCOTB_MK)/Makefile.sim
