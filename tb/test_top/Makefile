include ../common/cocotb.mk

VERILOG_SOURCES := $(abspath ../../rtl/pe.sv) \
                   $(abspath ../../rtl/systolic_array.sv) \
                   $(abspath ../../rtl/bias_relu.sv) \
                   $(abspath ../../rtl/requantize.sv) \
                   $(abspath ../../rtl/sram_wrapper.sv) \
                   $(abspath ../../rtl/ifm_feeder.sv) \
                   $(abspath ../../rtl/weight_loader.sv) \
                   $(abspath ../../rtl/ofm_writer.sv) \
                   $(abspath ../../rtl/apb_csr.sv) \
                   $(abspath ../../rtl/ctrl_fsm.sv) \
                   $(abspath ../../rtl/tinyNPU_top.sv) \
                   $(abspath top_harness.sv)
TOPLEVEL        := top_harness
MODULE          := test_top

include $(COCOTB_MK)/Makefile.sim
