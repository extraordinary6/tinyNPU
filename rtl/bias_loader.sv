// bias_loader.sv
// Reads one packed LANES*P_W-bit bias word from bias SRAM at base_addr and
// latches it to bias_out for the rest of the GEMM kick.
//
// FSM: IDLE -> FETCH -> LATCH -> DONE -> IDLE
//   FETCH : drives sram_en=1 with addr=base_addr (1 cycle of SRAM latency)
//   LATCH : sram_rdata is valid -> latch bias_out
//   DONE  : one-cycle done=1 pulse

module bias_loader #(
    parameter int LANES  = 4,
    parameter int P_W    = 32,
    parameter int ADDR_W = 12
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       start,
    input  logic [ADDR_W-1:0]          base_addr,

    output logic                       sram_en,
    output logic [ADDR_W-1:0]          sram_addr,
    input  logic [LANES*P_W-1:0]       sram_rdata,

    output logic [LANES*P_W-1:0]       bias_out,

    output logic                       busy,
    output logic                       done
);

    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_FETCH = 2'd1,
        S_LATCH = 2'd2,
        S_DONE  = 2'd3
    } state_t;

    state_t state, state_n;

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:  if (start) state_n = S_FETCH;
            S_FETCH:            state_n = S_LATCH;
            S_LATCH:            state_n = S_DONE;
            S_DONE:             state_n = S_IDLE;
            default:            state_n = S_IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n)                 bias_out <= {LANES*P_W{1'b0}};
        else if (state == S_LATCH)  bias_out <= sram_rdata;
    end

    assign sram_en   = (state == S_FETCH);
    assign sram_addr = base_addr;
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
