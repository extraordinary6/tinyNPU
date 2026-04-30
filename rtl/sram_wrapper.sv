// sram_wrapper.sv
// Behavioral single-port synchronous SRAM with 1-cycle read latency.
//   en=1 we=1 : mem[addr] <= wdata
//   en=1 we=0 : rdata     <= mem[addr]   (next-cycle visible)
//   en=0      : memory and rdata both hold
//
// Used by ifm_feeder, weight_loader, ofm_writer testbenches as a stand-in
// for real SRAM IP.

module sram_wrapper #(
    parameter int DEPTH  = 256,
    parameter int DATA_W = 32,
    parameter int ADDR_W = $clog2(DEPTH)
)(
    input  logic                  clk,
    input  logic                  en,
    input  logic                  we,
    input  logic [ADDR_W-1:0]     addr,
    input  logic [DATA_W-1:0]     wdata,
    output logic [DATA_W-1:0]     rdata
);

    logic [DATA_W-1:0] mem [DEPTH];

    // Write port.
    always_ff @(posedge clk) begin
        if (en && we) mem[addr] <= wdata;
    end

    // Read port.
    always_ff @(posedge clk) begin
        if (en && !we) rdata <= mem[addr];
    end

endmodule
