// ctrl_fsm.sv
// Top-level orchestration FSM. Drives the weight_loader, ifm_feeder and
// ofm_writer in sequence; reports busy / done / err back to apb_csr.
//
// Sequence (one GEMM kick from start_pulse):
//   IDLE  -- start && params_ok      --> LOAD_W
//         -- start && !params_ok     --> ERR
//   LOAD_W   -- wl_done              --> COMPUTE
//   COMPUTE  -- if_done              --> WRITEBACK
//   WRITEBACK-- ow_done              --> DONE
//   DONE                             --> IDLE
//   ERR                              --> IDLE
//
// Sub-module starts are 1-cycle pulses derived from "edge into state":
//   <sub>_start = (state == S_<X>) && (state_d != S_<X>)
// where state_d is state delayed by one cycle. params_ok rejects M/N/K==0.

module ctrl_fsm (
    input  logic              clk,
    input  logic              rst_n,

    // From apb_csr.
    input  logic              start,         // 1-cycle pulse
    input  logic [31:0]       m_count,
    input  logic [31:0]       n_count,
    input  logic [31:0]       k_count,

    // Sub-module handshakes.
    output logic              wl_start,
    input  logic              wl_done,
    output logic              if_start,
    input  logic              if_done,
    output logic              ow_start,
    input  logic              ow_done,

    // Status to apb_csr.
    output logic              busy,
    output logic              done,
    output logic              err
);

    typedef enum logic [2:0] {
        S_IDLE      = 3'd0,
        S_LOAD_W    = 3'd1,
        S_COMPUTE   = 3'd2,
        S_WRITEBACK = 3'd3,
        S_DONE      = 3'd4,
        S_ERR       = 3'd5
    } state_t;

    state_t state, state_n, state_d;

    logic params_ok;
    assign params_ok = (m_count != 32'h0) && (n_count != 32'h0) && (k_count != 32'h0);

    // ----- state register -----
    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    // ----- delayed state for entry-edge detection -----
    always_ff @(posedge clk) begin
        if (!rst_n) state_d <= S_IDLE;
        else        state_d <= state;
    end

    // ----- next state -----
    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE: begin
                if (start) begin
                    if (params_ok) state_n = S_LOAD_W;
                    else           state_n = S_ERR;
                end
            end
            S_LOAD_W:    if (wl_done) state_n = S_COMPUTE;
            S_COMPUTE:   if (if_done) state_n = S_WRITEBACK;
            S_WRITEBACK: if (ow_done) state_n = S_DONE;
            S_DONE:                   state_n = S_IDLE;
            S_ERR:                    state_n = S_IDLE;
            default:                  state_n = S_IDLE;
        endcase
    end

    // ----- sub-module start pulses (one cycle on entry into state) -----
    assign wl_start = (state == S_LOAD_W)    && (state_d != S_LOAD_W);
    assign if_start = (state == S_COMPUTE)   && (state_d != S_COMPUTE);
    assign ow_start = (state == S_WRITEBACK) && (state_d != S_WRITEBACK);

    // ----- status -----
    assign busy = (state != S_IDLE);
    assign done = (state == S_DONE);
    assign err  = (state == S_ERR);

endmodule
