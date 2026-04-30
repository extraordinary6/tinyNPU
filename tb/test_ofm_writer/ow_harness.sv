// ow_harness.sv
// ofm_writer + sram_wrapper for cocotb verification.
// A backdoor read port (bd_*) lets cocotb read SRAM contents post-run.

module ow_harness #(
    parameter int LANES  = 4,
    parameter int O_W    = 8,
    parameter int ADDR_W = 8,
    parameter int M_W    = 16,
    parameter int D_BUS  = LANES*O_W
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // Writer controls.
    input  logic                  start,
    input  logic [M_W-1:0]        m_count,
    input  logic [ADDR_W-1:0]     base_addr,
    input  logic [D_BUS-1:0]      data_in,
    input  logic                  data_valid,
    output logic                  busy,
    output logic                  done,

    // Backdoor read port (cocotb-driven SRAM reads to verify written data).
    input  logic                  bd_re,
    input  logic [ADDR_W-1:0]     bd_addr,
    output logic [D_BUS-1:0]      bd_rdata
);

    logic              ow_we;
    logic [ADDR_W-1:0] ow_addr;
    logic [D_BUS-1:0]  ow_wdata;

    logic              s_en;
    logic              s_we;
    logic [ADDR_W-1:0] s_addr;
    logic [D_BUS-1:0]  s_wdata;

    assign s_en    = ow_we ? 1'b1    : bd_re;
    assign s_we    = ow_we;
    assign s_addr  = ow_we ? ow_addr : bd_addr;
    assign s_wdata = ow_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(D_BUS),
        .ADDR_W(ADDR_W)
    ) u_sram (
        .clk   (clk),
        .en    (s_en),
        .we    (s_we),
        .addr  (s_addr),
        .wdata (s_wdata),
        .rdata (bd_rdata)
    );

    ofm_writer #(
        .LANES (LANES),
        .O_W   (O_W),
        .ADDR_W(ADDR_W),
        .M_W   (M_W)
    ) u_ow (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .m_count    (m_count),
        .base_addr  (base_addr),
        .data_in    (data_in),
        .data_valid (data_valid),
        .sram_we    (ow_we),
        .sram_addr  (ow_addr),
        .sram_wdata (ow_wdata),
        .busy       (busy),
        .done       (done)
    );

endmodule
