// pe.sv
// Single weight-stationary MAC cell for the 4x4 systolic array.
// Each cycle: psum_out_q <= psum_in + a_in * w_q; a_out_q <= a_in.
// Ports: clk, rst_n, w_load, w_in, a_in, a_out, psum_in, psum_out

module pe (
    input  logic                 clk,
    input  logic                 rst_n,

    // weight load (one-shot strobe; held when not loading)
    input  logic                 w_load,
    input  logic signed  [ 7:0]  w_in,

    // activation flow: left -> right
    input  logic signed  [ 7:0]  a_in,
    output logic signed  [ 7:0]  a_out,

    // partial-sum flow: top -> bottom
    input  logic signed  [31:0]  psum_in,
    output logic signed  [31:0]  psum_out
);

    logic signed [ 7:0]  w_q;
    logic signed [ 7:0]  a_q;
    logic signed [31:0]  psum_q;

    // Weight register: latched while w_load is asserted.
    always_ff @(posedge clk) begin
        if (!rst_n)      w_q <= 8'sd0;
        else if (w_load) w_q <= w_in;
    end

    // Activation pipeline register (1-cycle delay to right neighbor).
    always_ff @(posedge clk) begin
        if (!rst_n) a_q <= 8'sd0;
        else        a_q <= a_in;
    end

    // Partial-sum register: psum_in + a_in * w_q (signed).
    // SV signed promotion: 8s * 8s -> 16s, then +32s extends to 32s.
    always_ff @(posedge clk) begin
        if (!rst_n) psum_q <= 32'sd0;
        else        psum_q <= psum_in + a_in * w_q;
    end

    assign a_out    = a_q;
    assign psum_out = psum_q;

endmodule
