include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/ifm_feeder.sv) \
                   $(abspath if_harness.sv)
TOPLEVEL        := if_harness
MODULE          := test_ifm_feeder

include $(COCOTB_MK)/Makefile.sim
