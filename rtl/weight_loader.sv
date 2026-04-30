// weight_loader.sv
// Reads one packed weight tile (ROWS*COLS*A_W bits) from W SRAM at base_addr
// and emits a single-cycle w_load pulse with w_out for the systolic array.
//
// FSM: IDLE -> FETCH -> LOAD -> DONE -> IDLE
//   FETCH: drives sram_en=1 with addr=base_addr (1 cycle of SRAM latency)
//   LOAD : sram_rdata is valid -> w_load=1, w_out=sram_rdata
//   DONE : one-cycle done=1 pulse

module weight_loader #(
    parameter int ROWS   = 4,
    parameter int COLS   = 4,
    parameter int A_W    = 8,
    parameter int ADDR_W = 12
)(
    input  logic                              clk,
    input  logic                              rst_n,

    input  logic                              start,
    input  logic [ADDR_W-1:0]                 base_addr,

    output logic                              sram_en,
    output logic [ADDR_W-1:0]                 sram_addr,
    input  logic [ROWS*COLS*A_W-1:0]          sram_rdata,

    output logic                              w_load,
    output logic [ROWS*COLS*A_W-1:0]          w_out,

    output logic                              busy,
    output logic                              done
);

    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_FETCH = 2'd1,
        S_LOAD  = 2'd2,
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
            S_FETCH:            state_n = S_LOAD;
            S_LOAD:             state_n = S_DONE;
            S_DONE:             state_n = S_IDLE;
            default:            state_n = S_IDLE;
        endcase
    end

    assign sram_en   = (state == S_FETCH);
    assign sram_addr = base_addr;
    assign w_out     = sram_rdata;
    assign w_load    = (state == S_LOAD);
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
