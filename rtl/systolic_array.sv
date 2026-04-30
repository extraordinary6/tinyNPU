// systolic_array.sv
// 4x4 weight-stationary systolic MAC array, parameterized.
// Activations flow left->right, partial sums flow top->bottom.
// Top-row psum_in is hard-wired to 0; bottom-row psum exits via psum_out.
// w_load=1 latches all ROWS*COLS weights from w_in into the PE weight regs.
//
// Bus layout (LSB first, row-major):
//   w_in [(r*COLS + c)*A_W +: A_W]   -> weight at PE[r][c]
//   a_in [r*A_W +: A_W]              -> activation feeding row r (left side)
//   psum_out[c*P_W +: P_W]           -> partial sum exiting column c (bottom)

module systolic_array #(
    parameter int ROWS = 4,
    parameter int COLS = 4,
    parameter int A_W  = 8,
    parameter int P_W  = 32
)(
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic                              w_load,
    input  logic [ROWS*COLS*A_W-1:0]          w_in,

    input  logic [ROWS*A_W-1:0]               a_in,
    output logic [COLS*P_W-1:0]               psum_out
);

    // a_link[r][c]   : activation entering PE[r][c] from the left
    // a_link[r][COLS]: activation exiting PE[r][COLS-1] to the right (unused)
    // p_link[r][c]   : psum entering PE[r][c] from above
    // p_link[ROWS][c]: psum exiting PE[ROWS-1][c] downward
    logic signed [A_W-1:0]  a_link [ROWS][COLS+1];
    logic signed [P_W-1:0]  p_link [ROWS+1][COLS];

    genvar r, c;
    generate
        // Drive left edge of each row from the packed a_in bus.
        for (r = 0; r < ROWS; r++) begin : g_a_left
            assign a_link[r][0] = $signed(a_in[r*A_W +: A_W]);
        end

        // Top edge psum is zero.
        for (c = 0; c < COLS; c++) begin : g_p_top
            assign p_link[0][c] = {P_W{1'b0}};
        end

        // PE lattice.
        for (r = 0; r < ROWS; r++) begin : g_row
            for (c = 0; c < COLS; c++) begin : g_col
                logic signed [A_W-1:0] w_pe;
                assign w_pe = $signed(w_in[(r*COLS + c)*A_W +: A_W]);

                pe u_pe (
                    .clk     (clk),
                    .rst_n   (rst_n),
                    .w_load  (w_load),
                    .w_in    (w_pe),
                    .a_in    (a_link[r][c]),
                    .a_out   (a_link[r][c+1]),
                    .psum_in (p_link[r][c]),
                    .psum_out(p_link[r+1][c])
                );
            end
        end

        // Pack bottom row psums into output bus.
        for (c = 0; c < COLS; c++) begin : g_p_out
            assign psum_out[c*P_W +: P_W] = p_link[ROWS][c];
        end
    endgenerate

endmodule
