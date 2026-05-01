// req_param_loader.sv
// Loads per-lane requantize mult and shift parameters from W SRAM.

module req_param_loader #(
    parameter int LANES  = 4,
    parameter int P_W    = 32,
    parameter int ADDR_W = 12
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       start,
    input  logic [ADDR_W-1:0]          mult_base_addr,
    input  logic [ADDR_W-1:0]          shift_base_addr,

    output logic                       sram_en,
    output logic [ADDR_W-1:0]          sram_addr,
    input  logic [LANES*P_W-1:0]       sram_rdata,

    output logic [LANES*P_W-1:0]       mult_out,
    output logic [LANES*6-1:0]         shift_out,

    output logic                       busy,
    output logic                       done
);

    typedef enum logic [2:0] {
        S_IDLE        = 3'd0,
        S_FETCH_MULT  = 3'd1,
        S_LATCH_MULT  = 3'd2,
        S_FETCH_SHIFT = 3'd3,
        S_LATCH_SHIFT = 3'd4,
        S_DONE        = 3'd5
    } state_t;

    state_t state, state_n;

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:        if (start) state_n = S_FETCH_MULT;
            S_FETCH_MULT:             state_n = S_LATCH_MULT;
            S_LATCH_MULT:             state_n = S_FETCH_SHIFT;
            S_FETCH_SHIFT:            state_n = S_LATCH_SHIFT;
            S_LATCH_SHIFT:            state_n = S_DONE;
            S_DONE:                   state_n = S_IDLE;
            default:                  state_n = S_IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n)                       mult_out <= {LANES*P_W{1'b0}};
        else if (state == S_LATCH_MULT)   mult_out <= sram_rdata;
    end

    genvar c;
    generate
        for (c = 0; c < LANES; c++) begin : g_shift
            always_ff @(posedge clk) begin
                if (!rst_n)                         shift_out[c*6 +: 6] <= 6'd0;
                else if (state == S_LATCH_SHIFT)    shift_out[c*6 +: 6] <= sram_rdata[c*P_W +: 6];
            end
        end
    endgenerate

    assign sram_en   = (state == S_FETCH_MULT) || (state == S_FETCH_SHIFT);
    assign sram_addr = (state == S_FETCH_SHIFT) ? shift_base_addr : mult_base_addr;
    assign busy      = (state != S_IDLE);
    assign done      = (state == S_DONE);

endmodule
