// top_harness_8x8.sv
// Variant of top_harness with ROWS=COLS=8. Wraps tinyNPU_top together with
// four sram_wrapper instances (IFM/W/BIAS/OFM) and exposes backdoor write
// ports for IFM/W/BIAS plus a backdoor read port for OFM. Identical logic
// to top_harness.sv, just with the wider PE-array configuration.

module top_harness_8x8 #(
    parameter int ROWS   = 8,
    parameter int COLS   = 8,
    parameter int A_W    = 8,
    parameter int O_W    = 8,
    parameter int P_W    = 32,
    parameter int ADDR_W = 8,
    parameter int M_W    = 16
)(
    input  logic                          pclk,
    input  logic                          presetn,

    input  logic                          psel,
    input  logic                          penable,
    input  logic                          pwrite,
    input  logic [11:0]                   paddr,
    input  logic [31:0]                   pwdata,
    output logic [31:0]                   prdata,
    output logic                          pready,
    output logic                          pslverr,

    // IFM backdoor write port.
    input  logic                          bd_ifm_we,
    input  logic [ADDR_W-1:0]             bd_ifm_addr,
    input  logic [ROWS*A_W-1:0]           bd_ifm_wdata,

    // W backdoor write port.
    input  logic                          bd_w_we,
    input  logic [ADDR_W-1:0]             bd_w_addr,
    input  logic [ROWS*COLS*A_W-1:0]      bd_w_wdata,

    // BIAS backdoor write port.
    input  logic                          bd_bias_we,
    input  logic [ADDR_W-1:0]             bd_bias_addr,
    input  logic [COLS*P_W-1:0]           bd_bias_wdata,

    // OFM backdoor read port.
    input  logic                          bd_ofm_re,
    input  logic [ADDR_W-1:0]             bd_ofm_addr,
    output logic [COLS*O_W-1:0]           bd_ofm_rdata
);

    // ---------------- Top under test ----------------
    logic                       top_ifm_en;
    logic [ADDR_W-1:0]          top_ifm_addr;
    logic [ROWS*A_W-1:0]        top_ifm_rdata;

    logic                       top_w_en;
    logic [ADDR_W-1:0]          top_w_addr;
    logic [ROWS*COLS*A_W-1:0]   top_w_rdata;

    logic                       top_bias_en;
    logic [ADDR_W-1:0]          top_bias_addr;
    logic [COLS*P_W-1:0]        top_bias_rdata;

    logic                       top_ofm_we;
    logic [ADDR_W-1:0]          top_ofm_addr;
    logic [COLS*O_W-1:0]        top_ofm_wdata;

    tinyNPU_top #(
        .ROWS  (ROWS),
        .COLS  (COLS),
        .A_W   (A_W),
        .O_W   (O_W),
        .P_W   (P_W),
        .ADDR_W(ADDR_W),
        .M_W   (M_W)
    ) u_dut (
        .pclk            (pclk),
        .presetn         (presetn),
        .psel            (psel),
        .penable         (penable),
        .pwrite          (pwrite),
        .paddr           (paddr),
        .pwdata          (pwdata),
        .prdata          (prdata),
        .pready          (pready),
        .pslverr         (pslverr),
        .ifm_sram_en     (top_ifm_en),
        .ifm_sram_addr   (top_ifm_addr),
        .ifm_sram_rdata  (top_ifm_rdata),
        .w_sram_en       (top_w_en),
        .w_sram_addr     (top_w_addr),
        .w_sram_rdata    (top_w_rdata),
        .bias_sram_en    (top_bias_en),
        .bias_sram_addr  (top_bias_addr),
        .bias_sram_rdata (top_bias_rdata),
        .ofm_sram_we     (top_ofm_we),
        .ofm_sram_addr   (top_ofm_addr),
        .ofm_sram_wdata  (top_ofm_wdata)
    );

    // ---------------- IFM SRAM (mux backdoor write vs top read) ----------------
    logic                  ifm_en;
    logic                  ifm_we;
    logic [ADDR_W-1:0]     ifm_addr;
    logic [ROWS*A_W-1:0]   ifm_wdata;

    assign ifm_en    = bd_ifm_we ? 1'b1         : top_ifm_en;
    assign ifm_we    = bd_ifm_we;
    assign ifm_addr  = bd_ifm_we ? bd_ifm_addr  : top_ifm_addr;
    assign ifm_wdata = bd_ifm_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(ROWS*A_W),
        .ADDR_W(ADDR_W)
    ) u_ifm_sram (
        .clk   (pclk),
        .en    (ifm_en),
        .we    (ifm_we),
        .addr  (ifm_addr),
        .wdata (ifm_wdata),
        .rdata (top_ifm_rdata)
    );

    // ---------------- W SRAM ----------------
    logic                       w_en;
    logic                       w_we;
    logic [ADDR_W-1:0]          w_addr;
    logic [ROWS*COLS*A_W-1:0]   w_wdata;

    assign w_en    = bd_w_we ? 1'b1       : top_w_en;
    assign w_we    = bd_w_we;
    assign w_addr  = bd_w_we ? bd_w_addr  : top_w_addr;
    assign w_wdata = bd_w_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(ROWS*COLS*A_W),
        .ADDR_W(ADDR_W)
    ) u_w_sram (
        .clk   (pclk),
        .en    (w_en),
        .we    (w_we),
        .addr  (w_addr),
        .wdata (w_wdata),
        .rdata (top_w_rdata)
    );

    // ---------------- BIAS SRAM ----------------
    logic                  bs_en;
    logic                  bs_we;
    logic [ADDR_W-1:0]     bs_addr;
    logic [COLS*P_W-1:0]   bs_wdata;

    assign bs_en    = bd_bias_we ? 1'b1         : top_bias_en;
    assign bs_we    = bd_bias_we;
    assign bs_addr  = bd_bias_we ? bd_bias_addr : top_bias_addr;
    assign bs_wdata = bd_bias_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(COLS*P_W),
        .ADDR_W(ADDR_W)
    ) u_bias_sram (
        .clk   (pclk),
        .en    (bs_en),
        .we    (bs_we),
        .addr  (bs_addr),
        .wdata (bs_wdata),
        .rdata (top_bias_rdata)
    );

    // ---------------- OFM SRAM (mux backdoor read vs top write) ----------------
    logic                  ofm_en;
    logic                  ofm_we;
    logic [ADDR_W-1:0]     ofm_addr;
    logic [COLS*O_W-1:0]   ofm_wdata;

    assign ofm_en    = top_ofm_we ? 1'b1          : bd_ofm_re;
    assign ofm_we    = top_ofm_we;
    assign ofm_addr  = top_ofm_we ? top_ofm_addr  : bd_ofm_addr;
    assign ofm_wdata = top_ofm_wdata;

    sram_wrapper #(
        .DEPTH (1 << ADDR_W),
        .DATA_W(COLS*O_W),
        .ADDR_W(ADDR_W)
    ) u_ofm_sram (
        .clk   (pclk),
        .en    (ofm_en),
        .we    (ofm_we),
        .addr  (ofm_addr),
        .wdata (ofm_wdata),
        .rdata (bd_ofm_rdata)
    );

endmodule
