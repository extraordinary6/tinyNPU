include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/req_param_loader.sv) \
                   $(abspath rp_harness.sv)
TOPLEVEL        := rp_harness
MODULE          := test_req_param_loader

include $(COCOTB_MK)/Makefile.sim
