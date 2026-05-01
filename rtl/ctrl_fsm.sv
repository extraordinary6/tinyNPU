// ctrl_fsm.sv
// Top-level orchestration FSM. Supports K-tile accumulation: one kick runs
// k_tiles_total back-to-back tiles, each tile re-loading W and computing a
// partial GEMM that the row_accumulator (in top) sums across tiles.
//
// IDLE -> [params_ok? LOAD_W : ERR]
// LOAD_W -> [first_tile && bias_en?    LOAD_BIAS :
//            first_tile && pch_req_en? LOAD_REQ  :
//                                       COMPUTE]
// LOAD_BIAS -> [pch_req_en? LOAD_REQ : COMPUTE]
// LOAD_REQ  -> COMPUTE
// COMPUTE   -> [last_tile?  WRITEBACK : LOAD_W]   <-- tile loop
// WRITEBACK -> DONE -> IDLE  (or ERR -> IDLE)
//
// tile_idx increments at every COMPUTE -> LOAD_W transition (i.e., at the
// boundary between non-final tiles). first_tile = (tile_idx == 0).
// last_tile  = (tile_idx == k_tiles_total - 1).

module ctrl_fsm #(
    parameter int K_TILE_W = 8
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic                    start,
    input  logic [31:0]             m_count,
    input  logic [31:0]             n_count,
    input  logic [31:0]             k_count,
    input  logic [K_TILE_W-1:0]     k_tiles_total,
    input  logic                    bias_en,
    input  logic                    pch_req_en,

    output logic                    wl_start,
    input  logic                    wl_done,
    output logic                    bl_start,
    input  logic                    bl_done,
    output logic                    rp_start,
    input  logic                    rp_done,
    output logic                    if_start,
    input  logic                    if_done,
    output logic                    ow_start,
    input  logic                    ow_done,

    output logic [K_TILE_W-1:0]     tile_idx,
    output logic                    first_tile,
    output logic                    last_tile,

    output logic                    busy,
    output logic                    done,
    output logic                    err
);

    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_LOAD_W    = 3'd1,
        S_LOAD_BIAS = 3'd2,
        S_LOAD_REQ  = 3'd3,
        S_COMPUTE   = 3'd4,
        S_WRITEBACK = 3'd5,
        S_DONE      = 3'd6,
        S_ERR       = 3'd7
    } state_t;

    state_t state, state_n, state_d;
    logic [K_TILE_W-1:0] tile_idx_q;

    logic params_ok;
    assign params_ok = (m_count != 32'h0) && (n_count != 32'h0)
                       && (k_count != 32'h0) && (k_tiles_total != {K_TILE_W{1'b0}});

    assign tile_idx   = tile_idx_q;
    assign first_tile = (tile_idx_q == {K_TILE_W{1'b0}});
    assign last_tile  = (tile_idx_q == k_tiles_total - {{(K_TILE_W-1){1'b0}}, 1'b1});

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) state_d <= S_IDLE;
        else        state_d <= state;
    end

    always_ff @(posedge clk) begin
        if (!rst_n)                                              tile_idx_q <= {K_TILE_W{1'b0}};
        else if (state == S_IDLE)                                tile_idx_q <= {K_TILE_W{1'b0}};
        else if ((state == S_COMPUTE) && if_done && !last_tile)  tile_idx_q <= tile_idx_q + {{(K_TILE_W-1){1'b0}}, 1'b1};
    end

    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE: begin
                if (start) begin
                    if (params_ok) state_n = S_LOAD_W;
                    else           state_n = S_ERR;
                end
            end
            S_LOAD_W: begin
                if (wl_done) begin
                    if (first_tile && bias_en)         state_n = S_LOAD_BIAS;
                    else if (first_tile && pch_req_en) state_n = S_LOAD_REQ;
                    else                                state_n = S_COMPUTE;
                end
            end
            S_LOAD_BIAS: begin
                if (bl_done) begin
                    if (pch_req_en) state_n = S_LOAD_REQ;
                    else            state_n = S_COMPUTE;
                end
            end
            S_LOAD_REQ:  if (rp_done) state_n = S_COMPUTE;
            S_COMPUTE: begin
                if (if_done) begin
                    if (last_tile) state_n = S_WRITEBACK;
                    else           state_n = S_LOAD_W;
                end
            end
            S_WRITEBACK: if (ow_done) state_n = S_DONE;
            S_DONE:                   state_n = S_IDLE;
            S_ERR:                    state_n = S_IDLE;
            default:                  state_n = S_IDLE;
        endcase
    end

    assign wl_start = (state == S_LOAD_W)    && (state_d != S_LOAD_W);
    assign bl_start = (state == S_LOAD_BIAS) && (state_d != S_LOAD_BIAS);
    assign rp_start = (state == S_LOAD_REQ)  && (state_d != S_LOAD_REQ);
    assign if_start = (state == S_COMPUTE)   && (state_d != S_COMPUTE);
    assign ow_start = (state == S_WRITEBACK) && (state_d != S_WRITEBACK);

    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign err  = (state == S_ERR);

endmodule
