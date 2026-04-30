include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/pe.sv) $(abspath ../../rtl/systolic_array.sv)
TOPLEVEL        := systolic_array
MODULE          := test_systolic_array

include $(COCOTB_MK)/Makefile.sim
