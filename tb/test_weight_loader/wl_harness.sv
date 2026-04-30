// wl_harness.sv
// Glues weight_loader to a sram_wrapper instance for the cocotb DUT.
// Backdoor write port lets cocotb pre-load weights through the normal SRAM port.

module wl_harness #(
    parameter int ROWS   = 4,
    parameter int COLS   = 4,
    parameter int A_W    = 8,
    parameter int ADDR_W = 8,
    parameter int W_BUS  = ROWS*COLS*A_W
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  bd_we,
    input  logic [ADDR_W-1:0]     bd_addr,
    input  logic [W_BUS-1:0]      bd_wdata,

    input  logic                  start,
    input  logic [ADDR_W-1:0]     base_addr,
    output logic                  w_load,
    output logic [W_BUS-1:0]      w_out,
    output logic                  busy,
    output logic                  done
);

    logic              wl_sram_en;
    logic [ADDR_W-1:0] wl_sram_addr;
    logic [W_BUS-1:0]  wl_sram_rdata;

    logic              s_en;
    logic              s_we;
    logic [ADDR_W-1:0] s_addr;
    logic [W_BUS-1:0]  s_wdata;

    assign s_en    = bd_we ? 1'b1     : wl_sram_en;
    assign s_we    = bd_we;
    assign s_addr  = bd_we ? bd_addr  : wl_sram_addr;
    assign s_wdata = bd_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(W_BUS),
        .ADDR_W(ADDR_W)
    ) u_sram (
        .clk   (clk),
        .en    (s_en),
        .we    (s_we),
        .addr  (s_addr),
        .wdata (s_wdata),
        .rdata (wl_sram_rdata)
    );

    weight_loader #(
        .ROWS  (ROWS),
        .COLS  (COLS),
        .A_W   (A_W),
        .ADDR_W(ADDR_W)
    ) u_wl (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .base_addr  (base_addr),
        .sram_en    (wl_sram_en),
        .sram_addr  (wl_sram_addr),
        .sram_rdata (wl_sram_rdata),
        .w_load     (w_load),
        .w_out      (w_out),
        .busy       (busy),
        .done       (done)
    );

endmodule
