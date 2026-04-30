// accumulator.sv
// Per-lane INT32 accumulator for K-tile accumulation across passes of the
// systolic array. Synchronous reset 'rst_n', synchronous clear 'clr'
// (priority over en), per-lane enable bits.
//
// Behavior (each lane c, every clk):
//   if (!rst_n)        acc_q[c] <= 0
//   else if (clr)      acc_q[c] <= 0
//   else if (en[c])    acc_q[c] <= acc_q[c] + psum_in[c]
//
// Bus layout: psum_in / acc_out are flat packed, lane c at bits [c*P_W +: P_W].

module accumulator #(
    parameter int LANES = 4,
    parameter int P_W   = 32
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       clr,
    input  logic [LANES-1:0]           en,

    input  logic [LANES*P_W-1:0]       psum_in,
    output logic [LANES*P_W-1:0]       acc_out
);

    logic signed [P_W-1:0]  acc_q [LANES];

    genvar c;
    generate
        for (c = 0; c < LANES; c++) begin : g_lane
            logic signed [P_W-1:0] psum_lane;
            assign psum_lane = $signed(psum_in[c*P_W +: P_W]);

            always_ff @(posedge clk) begin
                if (!rst_n)     acc_q[c] <= {P_W{1'b0}};
                else if (clr)   acc_q[c] <= {P_W{1'b0}};
                else if (en[c]) acc_q[c] <= acc_q[c] + psum_lane;
            end

            assign acc_out[c*P_W +: P_W] = acc_q[c];
        end
    endgenerate

endmodule
