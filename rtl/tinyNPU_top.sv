// tinyNPU_top.sv
// Top-level integration. APB3-controlled GEMM with optional ReLU + requantize.
//
// Datapath: ifm_feeder -> systolic_array -> unskew -> bias_relu -> requantize
//           -> ofm_writer.
// Control:  apb_csr (CSR file) drives ctrl_fsm; ctrl_fsm sequences
//           weight_loader (LOAD_W) and then ifm_feeder + ofm_writer (COMPUTE
//           starts both; WRITEBACK waits for ow_done).
//
// v1 limitations:
//   * K is fixed to KCOLS (=4); no K-tile accumulation (accumulator is
//     bypassed). plan.md phase 7 extends this.
//   * bias_in is hard-wired to 0; FLAGS.BIAS_EN currently has no effect.
//
// Pipeline latency from if_start to first data_valid into ofm_writer = 9:
//   FSM startup + SRAM read(2) + systolic propagation(4) + unskew(3) = 9.

module tinyNPU_top #(
    parameter int ROWS   = 4,
    parameter int COLS   = 4,
    parameter int A_W    = 8,
    parameter int O_W    = 8,
    parameter int P_W    = 32,
    parameter int ADDR_W = 12,
    parameter int M_W    = 16
)(
    input  logic                          pclk,
    input  logic                          presetn,

    input  logic                          psel,
    input  logic                          penable,
    input  logic                          pwrite,
    input  logic [11:0]                   paddr,
    input  logic [31:0]                   pwdata,
    output logic [31:0]                   prdata,
    output logic                          pready,
    output logic                          pslverr,

    output logic                          ifm_sram_en,
    output logic [ADDR_W-1:0]             ifm_sram_addr,
    input  logic [ROWS*A_W-1:0]           ifm_sram_rdata,

    output logic                          w_sram_en,
    output logic [ADDR_W-1:0]             w_sram_addr,
    input  logic [ROWS*COLS*A_W-1:0]      w_sram_rdata,

    output logic                          ofm_sram_we,
    output logic [ADDR_W-1:0]             ofm_sram_addr,
    output logic [COLS*O_W-1:0]           ofm_sram_wdata
);

    // ---------------- CSR ----------------
    logic        start_pulse;
    logic        busy, done, err;
    logic [31:0] m_count_w, n_count_w, k_count_w;
    logic [31:0] ifm_base_w, w_base_w, ofm_base_w, bias_base_w;
    logic [31:0] flags_w, req_mult_w, req_shift_w;

    apb_csr u_csr (
        .pclk        (pclk),
        .presetn     (presetn),
        .psel        (psel),
        .penable     (penable),
        .pwrite      (pwrite),
        .paddr       (paddr),
        .pwdata      (pwdata),
        .prdata      (prdata),
        .pready      (pready),
        .pslverr     (pslverr),
        .start_pulse (start_pulse),
        .busy        (busy),
        .done        (done),
        .err         (err),
        .m_count     (m_count_w),
        .n_count     (n_count_w),
        .k_count     (k_count_w),
        .ifm_base    (ifm_base_w),
        .w_base      (w_base_w),
        .ofm_base    (ofm_base_w),
        .bias_base   (bias_base_w),
        .flags       (flags_w),
        .req_mult    (req_mult_w),
        .req_shift   (req_shift_w)
    );

    // ---------------- ctrl_fsm ----------------
    logic wl_start, wl_done;
    logic if_start, if_done;
    logic ow_start_unused, ow_done;

    ctrl_fsm u_fsm (
        .clk      (pclk),
        .rst_n    (presetn),
        .start    (start_pulse),
        .m_count  (m_count_w),
        .n_count  (n_count_w),
        .k_count  (k_count_w),
        .wl_start (wl_start),
        .wl_done  (wl_done),
        .if_start (if_start),
        .if_done  (if_done),
        .ow_start (ow_start_unused),
        .ow_done  (ow_done),
        .busy     (busy),
        .done     (done),
        .err      (err)
    );

    // ---------------- weight_loader ----------------
    logic                            w_load_pulse;
    logic [ROWS*COLS*A_W-1:0]        w_tile;
    logic                            wl_busy_unused;

    weight_loader #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .A_W    (A_W),
        .ADDR_W (ADDR_W)
    ) u_wl (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (wl_start),
        .base_addr  (w_base_w[ADDR_W-1:0]),
        .sram_en    (w_sram_en),
        .sram_addr  (w_sram_addr),
        .sram_rdata (w_sram_rdata),
        .w_load     (w_load_pulse),
        .w_out      (w_tile),
        .busy       (wl_busy_unused),
        .done       (wl_done)
    );

    // ---------------- ifm_feeder ----------------
    logic [ROWS*A_W-1:0]   feeder_a;
    logic                  if_busy_unused;

    ifm_feeder #(
        .ROWS   (ROWS),
        .A_W    (A_W),
        .ADDR_W (ADDR_W),
        .M_W    (M_W)
    ) u_if (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (if_start),
        .m_count    (m_count_w[M_W-1:0]),
        .base_addr  (ifm_base_w[ADDR_W-1:0]),
        .sram_en    (ifm_sram_en),
        .sram_addr  (ifm_sram_addr),
        .sram_rdata (ifm_sram_rdata),
        .a_out      (feeder_a),
        .busy       (if_busy_unused),
        .done       (if_done)
    );

    // ---------------- systolic_array ----------------
    logic [COLS*P_W-1:0]   sys_psum;

    systolic_array #(
        .ROWS (ROWS),
        .COLS (COLS),
        .A_W  (A_W),
        .P_W  (P_W)
    ) u_sys (
        .clk      (pclk),
        .rst_n    (presetn),
        .w_load   (w_load_pulse),
        .w_in     (w_tile),
        .a_in     (feeder_a),
        .psum_out (sys_psum)
    );

    // ---------------- unskew (ROWS=4 hard-wired shift register chains) ----------------
    // lane 0: 3 stages, lane 1: 2 stages, lane 2: 1 stage, lane 3: passthrough.
    logic [P_W-1:0] sk0_s0, sk0_s1, sk0_s2;
    logic [P_W-1:0] sk1_s0, sk1_s1;
    logic [P_W-1:0] sk2_s0;

    always_ff @(posedge pclk) begin
        if (!presetn) sk0_s0 <= {P_W{1'b0}};
        else          sk0_s0 <= sys_psum[0*P_W +: P_W];
    end
    always_ff @(posedge pclk) begin
        if (!presetn) sk0_s1 <= {P_W{1'b0}};
        else          sk0_s1 <= sk0_s0;
    end
    always_ff @(posedge pclk) begin
        if (!presetn) sk0_s2 <= {P_W{1'b0}};
        else          sk0_s2 <= sk0_s1;
    end
    always_ff @(posedge pclk) begin
        if (!presetn) sk1_s0 <= {P_W{1'b0}};
        else          sk1_s0 <= sys_psum[1*P_W +: P_W];
    end
    always_ff @(posedge pclk) begin
        if (!presetn) sk1_s1 <= {P_W{1'b0}};
        else          sk1_s1 <= sk1_s0;
    end
    always_ff @(posedge pclk) begin
        if (!presetn) sk2_s0 <= {P_W{1'b0}};
        else          sk2_s0 <= sys_psum[2*P_W +: P_W];
    end

    logic [COLS*P_W-1:0] psum_unskewed;
    assign psum_unskewed[0*P_W +: P_W] = sk0_s2;
    assign psum_unskewed[1*P_W +: P_W] = sk1_s1;
    assign psum_unskewed[2*P_W +: P_W] = sk2_s0;
    assign psum_unskewed[3*P_W +: P_W] = sys_psum[3*P_W +: P_W];

    // ---------------- bias_relu (bias hard-wired to 0 in v1) ----------------
    logic [COLS*P_W-1:0] bias_zero;
    logic [COLS*P_W-1:0] br_out;

    assign bias_zero = {COLS*P_W{1'b0}};

    bias_relu #(
        .LANES (COLS),
        .P_W   (P_W)
    ) u_br (
        .bias_en  (flags_w[0]),
        .relu_en  (flags_w[1]),
        .acc_in   (psum_unskewed),
        .bias_in  (bias_zero),
        .data_out (br_out)
    );

    // ---------------- requantize ----------------
    logic [COLS*O_W-1:0] req_out;

    requantize #(
        .LANES (COLS),
        .P_W   (P_W),
        .O_W   (O_W)
    ) u_req (
        .req_en   (flags_w[2]),
        .mult     (req_mult_w),
        .shift    (req_shift_w[5:0]),
        .acc_in   (br_out),
        .data_out (req_out)
    );

    // ---------------- valid_gen FSM ----------------
    typedef enum logic [1:0] {
        VG_IDLE = 2'd0,
        VG_LAT  = 2'd1,
        VG_VAL  = 2'd2
    } vg_state_t;

    vg_state_t vg_state, vg_state_n;
    logic [3:0]      vg_lat_cnt;
    logic [M_W-1:0]  vg_val_cnt;

    always_ff @(posedge pclk) begin
        if (!presetn) vg_state <= VG_IDLE;
        else          vg_state <= vg_state_n;
    end

    always_comb begin
        vg_state_n = vg_state;
        unique case (vg_state)
            VG_IDLE: if (if_start) vg_state_n = VG_LAT;
            VG_LAT:  if (vg_lat_cnt == 4'd7) vg_state_n = VG_VAL;
            VG_VAL:  if (vg_val_cnt == m_count_w[M_W-1:0] - {{(M_W-1){1'b0}}, 1'b1})
                        vg_state_n = VG_IDLE;
            default: vg_state_n = VG_IDLE;
        endcase
    end

    always_ff @(posedge pclk) begin
        if (!presetn)                vg_lat_cnt <= 4'd0;
        else if (vg_state != VG_LAT) vg_lat_cnt <= 4'd0;
        else                         vg_lat_cnt <= vg_lat_cnt + 4'd1;
    end

    always_ff @(posedge pclk) begin
        if (!presetn)                vg_val_cnt <= {M_W{1'b0}};
        else if (vg_state != VG_VAL) vg_val_cnt <= {M_W{1'b0}};
        else                         vg_val_cnt <= vg_val_cnt + {{(M_W-1){1'b0}}, 1'b1};
    end

    logic data_valid;
    assign data_valid = (vg_state == VG_VAL);

    // ---------------- ofm_writer ----------------
    logic ow_busy_unused;

    ofm_writer #(
        .LANES  (COLS),
        .O_W    (O_W),
        .ADDR_W (ADDR_W),
        .M_W    (M_W)
    ) u_ow (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (if_start),
        .m_count    (m_count_w[M_W-1:0]),
        .base_addr  (ofm_base_w[ADDR_W-1:0]),
        .data_in    (req_out),
        .data_valid (data_valid),
        .sram_we    (ofm_sram_we),
        .sram_addr  (ofm_sram_addr),
        .sram_wdata (ofm_sram_wdata),
        .busy       (ow_busy_unused),
        .done       (ow_done)
    );

    // ---------------- intentionally unused signal taps ----------------
    // CSR registers are 32 bits per APB convention, but internal ADDR_W is
    // smaller, FLAGS uses only [2:0], REQ_SHIFT uses only [5:0]. v1 also has
    // no bias_loader, so bias_base_w is unused.
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused_ok;
    assign _unused_ok = &{1'b0,
                          ifm_base_w[31:ADDR_W],
                          w_base_w[31:ADDR_W],
                          ofm_base_w[31:ADDR_W],
                          bias_base_w,
                          flags_w[31:3],
                          req_shift_w[31:6]};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
