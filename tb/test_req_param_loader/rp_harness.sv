// rp_harness.sv
// Wraps req_param_loader with a sram_wrapper instance and a backdoor write
// port so cocotb can pre-load mult/shift words at chosen addresses.

module rp_harness #(
    parameter int LANES  = 4,
    parameter int P_W    = 32,
    parameter int ADDR_W = 8,
    parameter int W_BUS  = LANES*P_W
)(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic                  bd_we,
    input  logic [ADDR_W-1:0]     bd_addr,
    input  logic [W_BUS-1:0]      bd_wdata,

    input  logic                  start,
    input  logic [ADDR_W-1:0]     mult_base_addr,
    input  logic [ADDR_W-1:0]     shift_base_addr,
    output logic [W_BUS-1:0]      mult_out,
    output logic [LANES*6-1:0]    shift_out,
    output logic                  busy,
    output logic                  done
);

    logic              rp_sram_en;
    logic [ADDR_W-1:0] rp_sram_addr;
    logic [W_BUS-1:0]  rp_sram_rdata;

    logic              s_en;
    logic              s_we;
    logic [ADDR_W-1:0] s_addr;
    logic [W_BUS-1:0]  s_wdata;

    assign s_en    = bd_we ? 1'b1     : rp_sram_en;
    assign s_we    = bd_we;
    assign s_addr  = bd_we ? bd_addr  : rp_sram_addr;
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
        .rdata (rp_sram_rdata)
    );

    req_param_loader #(
        .LANES (LANES),
        .P_W   (P_W),
        .ADDR_W(ADDR_W)
    ) u_rp (
        .clk             (clk),
        .rst_n           (rst_n),
        .start           (start),
        .mult_base_addr  (mult_base_addr),
        .shift_base_addr (shift_base_addr),
        .sram_en         (rp_sram_en),
        .sram_addr       (rp_sram_addr),
        .sram_rdata      (rp_sram_rdata),
        .mult_out        (mult_out),
        .shift_out       (shift_out),
        .busy            (busy),
        .done            (done)
    );

endmodule
