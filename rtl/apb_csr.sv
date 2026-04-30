// apb_csr.sv
// APB3 slave with the CSR file from plan.md §3.3.
// Single-cycle reads and writes; pready=1 in ACCESS phase, pslverr=1 on
// unmapped addresses. CTRL[0] is W1S (write 1 to set, hardware self-clears
// after one cycle) and is exposed as start_pulse to ctrl_fsm. STATUS is
// driven combinationally from busy/done/err inputs.

module apb_csr #(
    parameter logic [31:0] ID_MAGIC = 32'h4E50_5500   // "NPU\0"
)(
    input  logic              pclk,
    input  logic              presetn,

    // APB3 slave
    input  logic              psel,
    input  logic              penable,
    input  logic              pwrite,
    input  logic [11:0]       paddr,
    input  logic [31:0]       pwdata,
    output logic [31:0]       prdata,
    output logic              pready,
    output logic              pslverr,

    // To/from ctrl_fsm
    output logic              start_pulse,
    input  logic              busy,
    input  logic              done,
    input  logic              err,

    // Configuration outputs (consumed by ctrl_fsm + datapath)
    output logic [31:0]       m_count,
    output logic [31:0]       n_count,
    output logic [31:0]       k_count,
    output logic [31:0]       ifm_base,
    output logic [31:0]       w_base,
    output logic [31:0]       ofm_base,
    output logic [31:0]       bias_base,
    output logic [31:0]       flags,
    output logic [31:0]       req_mult,
    output logic [31:0]       req_shift
);

    // Address map.
    localparam logic [11:0] A_ID        = 12'h000;
    localparam logic [11:0] A_CTRL      = 12'h004;
    localparam logic [11:0] A_STATUS    = 12'h008;
    localparam logic [11:0] A_M         = 12'h00C;
    localparam logic [11:0] A_N         = 12'h010;
    localparam logic [11:0] A_K         = 12'h014;
    localparam logic [11:0] A_IFM       = 12'h018;
    localparam logic [11:0] A_W         = 12'h01C;
    localparam logic [11:0] A_OFM       = 12'h020;
    localparam logic [11:0] A_BIAS      = 12'h024;
    localparam logic [11:0] A_FLAGS     = 12'h028;
    localparam logic [11:0] A_REQ_MULT  = 12'h02C;
    localparam logic [11:0] A_REQ_SHIFT = 12'h030;

    // ----- access decode -----
    logic apb_access;
    logic apb_write;
    logic addr_valid;

    assign apb_access = psel && penable;
    assign apb_write  = apb_access && pwrite;

    always_comb begin
        unique case (paddr)
            A_ID, A_CTRL, A_STATUS, A_M, A_N, A_K,
            A_IFM, A_W, A_OFM, A_BIAS,
            A_FLAGS, A_REQ_MULT, A_REQ_SHIFT: addr_valid = 1'b1;
            default:                          addr_valid = 1'b0;
        endcase
    end

    assign pready  = apb_access;
    assign pslverr = apb_access && !addr_valid;

    // ----- CTRL[0]: W1S start, self-clearing after one cycle -----
    logic start_q;
    always_ff @(posedge pclk) begin
        if (!presetn)                                       start_q <= 1'b0;
        else if (apb_write && (paddr == A_CTRL) && pwdata[0]) start_q <= 1'b1;
        else                                                start_q <= 1'b0;
    end
    assign start_pulse = start_q;

    // ----- ordinary RW32 registers (one always_ff per register) -----
    always_ff @(posedge pclk) begin
        if (!presetn)                            m_count <= 32'h0;
        else if (apb_write && (paddr == A_M))    m_count <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            n_count <= 32'h0;
        else if (apb_write && (paddr == A_N))    n_count <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            k_count <= 32'h0;
        else if (apb_write && (paddr == A_K))    k_count <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            ifm_base <= 32'h0;
        else if (apb_write && (paddr == A_IFM))  ifm_base <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            w_base <= 32'h0;
        else if (apb_write && (paddr == A_W))    w_base <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            ofm_base <= 32'h0;
        else if (apb_write && (paddr == A_OFM))  ofm_base <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                            bias_base <= 32'h0;
        else if (apb_write && (paddr == A_BIAS)) bias_base <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                              flags <= 32'h0;
        else if (apb_write && (paddr == A_FLAGS))  flags <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                                  req_mult <= 32'h0;
        else if (apb_write && (paddr == A_REQ_MULT))   req_mult <= pwdata;
    end
    always_ff @(posedge pclk) begin
        if (!presetn)                                  req_shift <= 32'h0;
        else if (apb_write && (paddr == A_REQ_SHIFT))  req_shift <= pwdata;
    end

    // ----- read mux -----
    always_comb begin
        unique case (paddr)
            A_ID:        prdata = ID_MAGIC;
            A_CTRL:      prdata = {31'b0, start_q};
            A_STATUS:    prdata = {29'b0, err, done, busy};
            A_M:         prdata = m_count;
            A_N:         prdata = n_count;
            A_K:         prdata = k_count;
            A_IFM:       prdata = ifm_base;
            A_W:         prdata = w_base;
            A_OFM:       prdata = ofm_base;
            A_BIAS:      prdata = bias_base;
            A_FLAGS:     prdata = flags;
            A_REQ_MULT:  prdata = req_mult;
            A_REQ_SHIFT: prdata = req_shift;
            default:     prdata = 32'h0;
        endcase
    end

endmodule
