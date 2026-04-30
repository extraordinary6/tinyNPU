// bias_relu.sv
// Per-lane combinational bias-add + ReLU stage.
//   sum  = bias_en ? acc + bias : acc
//   out  = (relu_en && sum<0) ? 0 : sum
//
// All lanes are independent. Bus layout (LSB-first): lane c at [c*P_W +: P_W].

module bias_relu #(
    parameter int LANES = 4,
    parameter int P_W   = 32
)(
    input  logic                       bias_en,
    input  logic                       relu_en,

    input  logic [LANES*P_W-1:0]       acc_in,
    input  logic [LANES*P_W-1:0]       bias_in,
    output logic [LANES*P_W-1:0]       data_out
);

    genvar c;
    generate
        for (c = 0; c < LANES; c++) begin : g_lane
            logic signed [P_W-1:0]  acc_lane;
            logic signed [P_W-1:0]  bias_lane;
            logic signed [P_W-1:0]  sum_lane;
            logic signed [P_W-1:0]  out_lane;

            assign acc_lane  = $signed(acc_in [c*P_W +: P_W]);
            assign bias_lane = $signed(bias_in[c*P_W +: P_W]);
            assign sum_lane  = bias_en ? (acc_lane + bias_lane) : acc_lane;
            assign out_lane  = (relu_en && sum_lane[P_W-1]) ? {P_W{1'b0}} : sum_lane;

            assign data_out[c*P_W +: P_W] = out_lane;
        end
    endgenerate

endmodule
