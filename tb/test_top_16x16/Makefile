include ../common/cocotb.mk

VERILOG_SOURCES := \
	$(abspath ../../rtl/pe.sv) \
	$(abspath ../../rtl/systolic_array.sv) \
	$(abspath ../../rtl/bias_relu.sv) \
	$(abspath ../../rtl/requantize.sv) \
	$(abspath ../../rtl/req_param_loader.sv) \
	$(abspath ../../rtl/bias_loader.sv) \
	$(abspath ../../rtl/row_accumulator.sv) \
	$(abspath ../../rtl/sram_wrapper.sv) \
	$(abspath ../../rtl/ifm_feeder.sv) \
	$(abspath ../../rtl/weight_loader.sv) \
	$(abspath ../../rtl/ofm_writer.sv) \
	$(abspath ../../rtl/apb_csr.sv) \
	$(abspath ../../rtl/ctrl_fsm.sv) \
	$(abspath ../../rtl/tinyNPU_top.sv) \
	$(abspath top_harness_16x16.sv)

TOPLEVEL := top_harness_16x16
MODULE   := test_top_16x16

include $(COCOTB_MK)/Makefile.sim
