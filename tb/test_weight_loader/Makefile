include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/weight_loader.sv) \
                   $(abspath wl_harness.sv)
TOPLEVEL        := wl_harness
MODULE          := test_weight_loader

include $(COCOTB_MK)/Makefile.sim
