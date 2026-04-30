// ofm_writer.sv
// Streams downstream results into OFM SRAM. Each accepted (data_valid=1) cycle
// during S_WRITE consumes one element and writes one SRAM word at base_addr+i.
//
// FSM:
//   IDLE  -- start --> WRITE
//   WRITE: counts accepted beats; transitions to DONE when the m_count-th beat
//          is accepted (i.e., write_cnt == m_count - 1 with data_valid=1).
//   DONE: one-cycle done=1 pulse, then back to IDLE.

module ofm_writer #(
    parameter int LANES  = 4,
    parameter int O_W    = 8,
    parameter int ADDR_W = 12,
    parameter int M_W    = 16
)(
    input  logic                       clk,
    input  logic                       rst_n,

    input  logic                       start,
    input  logic [M_W-1:0]             m_count,
    input  logic [ADDR_W-1:0]          base_addr,

    input  logic [LANES*O_W-1:0]       data_in,
    input  logic                       data_valid,

    output logic                       sram_we,
    output logic [ADDR_W-1:0]          sram_addr,
    output logic [LANES*O_W-1:0]       sram_wdata,

    output logic                       busy,
    output logic                       done
);

    typedef enum logic [1:0] {
        S_IDLE  = 2'd0,
        S_WRITE = 2'd1,
        S_DONE  = 2'd2
    } state_t;

    state_t state, state_n;
    logic [M_W-1:0] write_cnt;
    logic           accept;

    assign accept = (state == S_WRITE) && data_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= state_n;
    end

    always_comb begin
        state_n = state;
        unique case (state)
            S_IDLE:  if (start) state_n = S_WRITE;
            S_WRITE: if (accept && (write_cnt == m_count - {{(M_W-1){1'b0}}, 1'b1}))
                        state_n = S_DONE;
            S_DONE:  state_n = S_IDLE;
            default: state_n = S_IDLE;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n)               write_cnt <= {M_W{1'b0}};
        else if (state == S_IDLE) write_cnt <= {M_W{1'b0}};
        else if (accept)          write_cnt <= write_cnt + {{(M_W-1){1'b0}}, 1'b1};
    end

    assign sram_we    = accept;
    assign sram_addr  = base_addr + ADDR_W'(write_cnt);
    assign sram_wdata = data_in;
    assign busy       = (state != S_IDLE);
    assign done       = (state == S_DONE);

endmodule
