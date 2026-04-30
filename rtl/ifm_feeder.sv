// ifm_feeder.sv
// Reads a M-row IFM matrix from SRAM (one row of ROWS x INT8 per address)
// and produces the staggered activation stream that the systolic array
// expects: a_out[r] @ cycle (i + r + 1) == A[i, r]  (for 0 <= i < M).
//
// Pipeline structure (Icarus-friendly: no mixed proc/cont drivers per array):
//   pipe0          (comb)  : gated injection point = sram_en_d ? sram_rdata : 0
//   pipeR[0..R-2]  (regs)  : pipeR[0] <= pipe0 ; pipeR[s] <= pipeR[s-1]
// Lane 0 reads byte 0 of pipe0; lane r (r>=1) reads byte r of pipeR[r-1].
//
// FSM: IDLE -> READ -> DRAIN -> DONE -> IDLE.

module ifm_feeder #(
    parameter int ROWS   = 4,
    parameter int A_W    = 8,
    parameter int ADDR_W = 12,
    parameter int M_W    = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       start,
    input  logic [M_W-1:0]             m_count,
    input  logic [ADDR_W-1:0]          base_addr,

    output logic                       sram_en,
    output logic [ADDR_W-1:0]          sram_addr,
    input  logic [ROWS*A_W-1:0]        sram_rdata,

    output logic [ROWS*A_W-1:0]        a_out,

    output logic                       busy,
    output logic                       done
);

    localparam int DCW = $clog2(ROWS + 1);

    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_READ  = 2'd1,
        S_DRAIN = 2'd2,
        S_DONE  = 2'd3
    } state_t;

    state_t state, state_n;

    logic [M_W-1:0]   read_cnt;
    logic [DCW-1:0]   drain_cnt;
    logic             sram_en_d;

    // ----- state register -----
    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    // ----- next state -----
    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:  if (start)
                        state_n = S_READ;
            S_READ:  if (read_cnt == m_count - {{(M_W-1){1'b0}}, 1'b1})
                        state_n = S_DRAIN;
            S_DRAIN: if (drain_cnt == DCW'(ROWS - 1))
                        state_n = S_DONE;
            S_DONE:  state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    // ----- read_cnt -----
    always_ff @(posedge clk) begin
        if (!rst_n)               read_cnt <= {M_W{1'b0}};
        else if (state == S_IDLE) read_cnt <= {M_W{1'b0}};
        else if (state == S_READ) read_cnt <= read_cnt + {{(M_W-1){1'b0}}, 1'b1};
    end

    // ----- drain_cnt -----
    always_ff @(posedge clk) begin
        if (!rst_n)                 drain_cnt <= {DCW{1'b0}};
        else if (state != S_DRAIN)  drain_cnt <= {DCW{1'b0}};
        else                        drain_cnt <= drain_cnt + {{(DCW-1){1'b0}}, 1'b1};
    end

    // ----- sram_en delayed by 1 cycle (marks when sram_rdata is valid) -----
    always_ff @(posedge clk) begin
        if (!rst_n) sram_en_d <= 1'b0;
        else        sram_en_d <= sram_en;
    end

    // ----- SRAM control -----
    assign sram_en   = (state == S_READ);
    assign sram_addr = base_addr + ADDR_W'(read_cnt);

    // ----- stagger pipeline -----
    logic [ROWS*A_W-1:0] pipe0;
    logic [ROWS*A_W-1:0] pipeR [ROWS-1];
    logic                shift_en;

    assign shift_en = sram_en_d || (state == S_DRAIN);
    assign pipe0    = sram_en_d ? sram_rdata : {ROWS*A_W{1'b0}};

    genvar s;
    generate
        for (s = 0; s < ROWS - 1; s++) begin : g_pipe
            if (s == 0) begin : g_first
                always_ff @(posedge clk) begin
                    if (!rst_n)        pipeR[0] <= {ROWS*A_W{1'b0}};
                    else if (shift_en) pipeR[0] <= pipe0;
                end
            end else begin : g_rest
                always_ff @(posedge clk) begin
                    if (!rst_n)        pipeR[s]   <= {ROWS*A_W{1'b0}};
                    else if (shift_en) pipeR[s]   <= pipeR[s-1];
                end
            end
        end
    endgenerate

    // ----- output extraction -----
    assign a_out[0*A_W +: A_W] = pipe0[0*A_W +: A_W];
    genvar r;
    generate
        for (r = 1; r < ROWS; r++) begin : g_out
            assign a_out[r*A_W +: A_W] = pipeR[r-1][r*A_W +: A_W];
        end
    endgenerate

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);

endmodule
