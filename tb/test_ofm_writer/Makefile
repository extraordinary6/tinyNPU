include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/ofm_writer.sv) \
                   $(abspath ow_harness.sv)
TOPLEVEL        := ow_harness
MODULE          := test_ofm_writer

include $(COCOTB_MK)/Makefile.sim
