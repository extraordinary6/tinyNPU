// row_accumulator.sv
// Per-(row, lane) INT32 accumulator for K-tile GEMM. Stores up to M_MAX rows
// of LANES INT32 partial sums in a flat memory addressed by row_idx.
//
// On each data_valid cycle:
//   read_value = first_tile ? 0 : mem[row_idx]
//   sum_value  = read_value + psum_in     (per-lane signed add)
//   mem[row_idx] <= sum_value             (writeback when data_valid=1)
//   acc_out     = sum_value               (combinational, used downstream)
//
// On first_tile=1, mem[row_idx] is overwritten with psum_in directly (the
// previous tile-accumulated state is irrelevant). Memory has no reset; back
// -to-back kicks work because tile 0 always writes through with first_tile=1.

module row_accumulator #(
    parameter int LANES = 4,
    parameter int P_W   = 32,
    parameter int M_W   = 16,
    parameter int M_MAX = 64
)(
    input  logic                       clk,

    input  logic                       data_valid,
    input  logic [M_W-1:0]             row_idx,
    input  logic                       first_tile,
    input  logic [LANES*P_W-1:0]       psum_in,

    output logic [LANES*P_W-1:0]       acc_out
);

    localparam int RA_W = $clog2(M_MAX);

    logic [LANES*P_W-1:0] mem [M_MAX];
    logic [LANES*P_W-1:0] read_value;
    logic [LANES*P_W-1:0] sum_value;
    logic [RA_W-1:0]      idx;

    assign idx        = row_idx[RA_W-1:0];
    assign read_value = first_tile ? {LANES*P_W{1'b0}} : mem[idx];

    genvar c;
    generate
        for (c = 0; c < LANES; c++) begin : g_lane
            logic signed [P_W-1:0] r_lane;
            logic signed [P_W-1:0] p_lane;
            logic signed [P_W-1:0] s_lane;

            assign r_lane = $signed(read_value[c*P_W +: P_W]);
            assign p_lane = $signed(psum_in   [c*P_W +: P_W]);
            assign s_lane = r_lane + p_lane;

            assign sum_value[c*P_W +: P_W] = s_lane;
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (data_valid) mem[idx] <= sum_value;
    end

    assign acc_out = sum_value;

    // M_W high bits of row_idx are intentionally ignored (idx is RA_W bits).
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused_ok;
    assign _unused_ok = &{1'b0, row_idx[M_W-1:RA_W]};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
