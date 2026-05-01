// requantize.sv
// TFLite-lite requantize: INT32 -> INT8 via per-lane mult-shift-saturate.

module requantize #(
    parameter int LANES = 4,
    parameter int P_W   = 32,
    parameter int O_W   = 8
)(
    input  logic                       req_en,
    input  logic [LANES*P_W-1:0]       mult,
    input  logic [LANES*6-1:0]         shift,

    input  logic [LANES*P_W-1:0]       acc_in,
    output logic [LANES*O_W-1:0]       data_out
);

    genvar c;
    generate
        for (c = 0; c < LANES; c++) begin : g_lane
            logic signed [P_W-1:0]   acc_lane;
            logic signed [P_W-1:0]   mult_lane;
            logic        [5:0]       shift_lane;
            logic signed [63:0]      acc_ext;
            logic signed [63:0]      mult_ext;
            logic signed [63:0]      product;
            logic signed [63:0]      round_bias;
            logic signed [63:0]      shifted;
            logic signed [O_W-1:0]   sat_out;
            logic signed [O_W-1:0]   trunc_out;
            logic signed [O_W-1:0]   out_lane;

            assign acc_lane   = $signed(acc_in[c*P_W +: P_W]);
            assign mult_lane  = $signed(mult[c*P_W +: P_W]);
            assign shift_lane = shift[c*6 +: 6];
            assign acc_ext    = {{(64-P_W){acc_lane[P_W-1]}}, acc_lane};
            assign mult_ext   = {{(64-P_W){mult_lane[P_W-1]}}, mult_lane};
            assign product    = acc_ext * mult_ext;
            assign round_bias = (shift_lane == 6'd0)
                                ? 64'sd0
                                : (64'sd1 <<< (shift_lane - 6'd1));
            assign shifted    = $signed(product + round_bias) >>> shift_lane;
            assign sat_out    = (shifted >  64'sd127) ?  8'sd127 :
                                (shifted < -64'sd128) ? -8'sd128 :
                                                         shifted[O_W-1:0];
            assign trunc_out  = acc_lane[O_W-1:0];
            assign out_lane   = req_en ? sat_out : trunc_out;

            assign data_out[c*O_W +: O_W] = out_lane;
        end
    endgenerate

endmodule
