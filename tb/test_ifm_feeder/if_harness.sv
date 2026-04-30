// if_harness.sv
// ifm_feeder + sram_wrapper for cocotb verification.
// Backdoor write port lets cocotb pre-load IFM rows.

module if_harness #(
    parameter int ROWS   = 4,
    parameter int A_W    = 8,
    parameter int ADDR_W = 8,
    parameter int M_W    = 16,
    parameter int A_BUS  = ROWS*A_W
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  bd_we,
    input  logic [ADDR_W-1:0]     bd_addr,
    input  logic [A_BUS-1:0]      bd_wdata,

    input  logic                  start,
    input  logic [M_W-1:0]        m_count,
    input  logic [ADDR_W-1:0]     base_addr,
    output logic [A_BUS-1:0]      a_out,
    output logic                  busy,
    output logic                  done
);

    logic              if_sram_en;
    logic [ADDR_W-1:0] if_sram_addr;
    logic [A_BUS-1:0]  if_sram_rdata;

    logic              s_en;
    logic              s_we;
    logic [ADDR_W-1:0] s_addr;
    logic [A_BUS-1:0]  s_wdata;

    assign s_en    = bd_we ? 1'b1 : if_sram_en;
    assign s_we    = bd_we;
    assign s_addr  = bd_we ? bd_addr : if_sram_addr;
    assign s_wdata = bd_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(A_BUS),
        .ADDR_W(ADDR_W)
    ) u_sram (
        .clk   (clk),
        .en    (s_en),
        .we    (s_we),
        .addr  (s_addr),
        .wdata (s_wdata),
        .rdata (if_sram_rdata)
    );

    ifm_feeder #(
        .ROWS  (ROWS),
        .A_W   (A_W),
        .ADDR_W(ADDR_W),
        .M_W   (M_W)
    ) u_if (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .m_count    (m_count),
        .base_addr  (base_addr),
        .sram_en    (if_sram_en),
        .sram_addr  (if_sram_addr),
        .sram_rdata (if_sram_rdata),
        .a_out      (a_out),
        .busy       (busy),
        .done       (done)
    );

endmodule
