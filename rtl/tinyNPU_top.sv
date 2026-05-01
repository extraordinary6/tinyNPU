// tinyNPU_top.sv
// Top-level integration. APB3-controlled GEMM with optional bias / ReLU /
// (per-channel) requantize, and K-tile accumulation across multiple weight
// tiles.
//
// Datapath: ifm_feeder -> systolic_array -> unskew -> row_accumulator
//           -> bias_relu -> requantize -> ofm_writer.
// Control:  apb_csr (CSR file) drives ctrl_fsm; ctrl_fsm sequences
//           weight_loader (LOAD_W), optional bias_loader (LOAD_BIAS),
//           optional req_param_loader (LOAD_REQ), then ifm_feeder + ofm_writer.
//           For K > KCOLS=COLS, ctrl_fsm loops LOAD_W <-> COMPUTE
//           k_tiles_total = K/COLS times before WRITEBACK.
//
// SRAM layout (caller responsibility):
//   IFM SRAM: A[M, K] split tile-major. Tile k slice A[:, k*COLS:(k+1)*COLS]
//             stored at addresses ifm_base + k*M ... + (k+1)*M - 1.
//   W   SRAM: tile k weight slab B[k*COLS:(k+1)*COLS, 0:COLS] at w_base + k.
//   OFM SRAM: M output rows at ofm_base + 0 ... + M-1.
//
// Pipeline latency from if_start to first data_valid into ofm_writer = 9.

module tinyNPU_top #(
    parameter int ROWS   = 4,
    parameter int COLS   = 4,
    parameter int A_W    = 8,
    parameter int O_W    = 8,
    parameter int P_W    = 32,
    parameter int ADDR_W = 12,
    parameter int M_W    = 16,
    parameter int M_MAX  = 64,
    parameter int K_TILE_W = 8
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

    output logic                          bias_sram_en,
    output logic [ADDR_W-1:0]             bias_sram_addr,
    input  logic [COLS*P_W-1:0]           bias_sram_rdata,

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
    logic [31:0] req_mult_base_w, req_shift_base_w;

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
        .m_count        (m_count_w),
        .n_count        (n_count_w),
        .k_count        (k_count_w),
        .ifm_base       (ifm_base_w),
        .w_base         (w_base_w),
        .ofm_base       (ofm_base_w),
        .bias_base      (bias_base_w),
        .flags          (flags_w),
        .req_mult       (req_mult_w),
        .req_shift      (req_shift_w),
        .req_mult_base  (req_mult_base_w),
        .req_shift_base (req_shift_base_w)
    );

    // ---------------- ctrl_fsm ----------------
    logic                    wl_start, wl_done;
    logic                    bl_start, bl_done;
    logic                    rp_start, rp_done;
    logic                    if_start, if_done;
    logic                    ow_start_int, ow_done;
    logic [K_TILE_W-1:0]     tile_idx;
    logic                    first_tile, last_tile;
    logic [K_TILE_W-1:0]     k_tiles_total;

    // K must be a multiple of COLS; software is responsible. k_tiles_total =
    // k_count_w / COLS = k_count_w >> $clog2(COLS).
    localparam int LOG2_COLS = $clog2(COLS);
    assign k_tiles_total = k_count_w[LOG2_COLS +: K_TILE_W];

    ctrl_fsm #(
        .K_TILE_W (K_TILE_W)
    ) u_fsm (
        .clk           (pclk),
        .rst_n         (presetn),
        .start         (start_pulse),
        .m_count       (m_count_w),
        .n_count       (n_count_w),
        .k_count       (k_count_w),
        .k_tiles_total (k_tiles_total),
        .bias_en       (flags_w[0]),
        .pch_req_en    (flags_w[3]),
        .wl_start      (wl_start),
        .wl_done       (wl_done),
        .bl_start      (bl_start),
        .bl_done       (bl_done),
        .rp_start      (rp_start),
        .rp_done       (rp_done),
        .if_start      (if_start),
        .if_done       (if_done),
        .ow_start      (ow_start_int),
        .ow_done       (ow_done),
        .tile_idx      (tile_idx),
        .first_tile    (first_tile),
        .last_tile     (last_tile),
        .busy          (busy),
        .done          (done),
        .err           (err)
    );

    // ---------------- per-tile address computation ----------------
    // weight_loader: each tile is exactly one address (one packed ROWS*COLS*A_W word).
    logic [ADDR_W-1:0] w_base_tile;
    assign w_base_tile = w_base_w[ADDR_W-1:0]
                         + {{(ADDR_W-K_TILE_W){1'b0}}, tile_idx};

    // ifm_feeder: each tile occupies M consecutive addresses. Track an offset
    // register that resets at the start of a kick and advances by m_count
    // when entering a non-first tile's COMPUTE.
    logic [ADDR_W-1:0] ifm_base_tile;
    always_ff @(posedge pclk) begin
        if (!presetn)                           ifm_base_tile <= {ADDR_W{1'b0}};
        else if (start_pulse)                   ifm_base_tile <= ifm_base_w[ADDR_W-1:0];
        else if (if_start && !first_tile)       ifm_base_tile <= ifm_base_tile + m_count_w[ADDR_W-1:0];
    end

    // ---------------- weight_loader ----------------
    logic                            w_load_pulse;
    logic [ROWS*COLS*A_W-1:0]        w_tile;
    logic                            wl_busy_unused;
    logic                            wl_sram_en;
    logic [ADDR_W-1:0]               wl_sram_addr;
    logic                            rp_sram_en;
    logic [ADDR_W-1:0]               rp_sram_addr;

    weight_loader #(
        .ROWS   (ROWS),
        .COLS   (COLS),
        .A_W    (A_W),
        .ADDR_W (ADDR_W)
    ) u_wl (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (wl_start),
        .base_addr  (w_base_tile),
        .sram_en    (wl_sram_en),
        .sram_addr  (wl_sram_addr),
        .sram_rdata (w_sram_rdata),
        .w_load     (w_load_pulse),
        .w_out      (w_tile),
        .busy       (wl_busy_unused),
        .done       (wl_done)
    );

    logic [COLS*P_W-1:0] req_mult_loaded;
    logic [COLS*6-1:0]   req_shift_loaded;
    logic                rp_busy_unused;

    req_param_loader #(
        .LANES  (COLS),
        .P_W    (P_W),
        .ADDR_W (ADDR_W)
    ) u_rp (
        .clk             (pclk),
        .rst_n           (presetn),
        .start           (rp_start),
        .mult_base_addr  (req_mult_base_w[ADDR_W-1:0]),
        .shift_base_addr (req_shift_base_w[ADDR_W-1:0]),
        .sram_en         (rp_sram_en),
        .sram_addr       (rp_sram_addr),
        .sram_rdata      (w_sram_rdata),
        .mult_out        (req_mult_loaded),
        .shift_out       (req_shift_loaded),
        .busy            (rp_busy_unused),
        .done            (rp_done)
    );

    assign w_sram_en   = wl_sram_en || rp_sram_en;
    assign w_sram_addr = rp_sram_en ? rp_sram_addr : wl_sram_addr;

    // ---------------- bias_loader ----------------
    logic [COLS*P_W-1:0] bias_loaded;
    logic                bl_busy_unused;

    bias_loader #(
        .LANES  (COLS),
        .P_W    (P_W),
        .ADDR_W (ADDR_W)
    ) u_bl (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (bl_start),
        .base_addr  (bias_base_w[ADDR_W-1:0]),
        .sram_en    (bias_sram_en),
        .sram_addr  (bias_sram_addr),
        .sram_rdata (bias_sram_rdata),
        .bias_out   (bias_loaded),
        .busy       (bl_busy_unused),
        .done       (bl_done)
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
        .base_addr  (ifm_base_tile),
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

    // ---------------- valid_gen FSM (per-tile data_valid window) ----------------
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

    // ---------------- data-side tile counter ----------------
    // ctrl_fsm.first_tile flips on the if_done edge, but valid_gen's data
    // window for tile k can extend a few cycles past if_done (the unskew
    // tail). Track tile boundaries on the valid_gen side so the
    // row_accumulator sees first_tile=1 for the entire span of tile 0's
    // M-row data window.
    logic data_done_pulse;
    assign data_done_pulse = (vg_state == VG_VAL) && (vg_state_n == VG_IDLE);

    logic [K_TILE_W-1:0] data_tile_idx;
    always_ff @(posedge pclk) begin
        if (!presetn)             data_tile_idx <= {K_TILE_W{1'b0}};
        else if (start_pulse)     data_tile_idx <= {K_TILE_W{1'b0}};
        else if (data_done_pulse) data_tile_idx <= data_tile_idx + {{(K_TILE_W-1){1'b0}}, 1'b1};
    end

    logic data_first_tile;
    logic data_last_tile;
    assign data_first_tile = (data_tile_idx == {K_TILE_W{1'b0}});
    assign data_last_tile  = (data_tile_idx == k_tiles_total - {{(K_TILE_W-1){1'b0}}, 1'b1});

    // ---------------- row index for accumulator ----------------
    logic [M_W-1:0] row_idx;
    always_ff @(posedge pclk) begin
        if (!presetn)        row_idx <= {M_W{1'b0}};
        else if (if_start)   row_idx <= {M_W{1'b0}};
        else if (data_valid) row_idx <= row_idx + {{(M_W-1){1'b0}}, 1'b1};
    end

    // ---------------- row_accumulator (between unskew and bias_relu) ----------------
    logic [COLS*P_W-1:0] acc_out;

    row_accumulator #(
        .LANES (COLS),
        .P_W   (P_W),
        .M_W   (M_W),
        .M_MAX (M_MAX)
    ) u_acc (
        .clk        (pclk),
        .data_valid (data_valid),
        .row_idx    (row_idx),
        .first_tile (data_first_tile),
        .psum_in    (psum_unskewed),
        .acc_out    (acc_out)
    );

    // ---------------- bias_relu ----------------
    logic [COLS*P_W-1:0] br_out;

    bias_relu #(
        .LANES (COLS),
        .P_W   (P_W)
    ) u_br (
        .bias_en  (flags_w[0]),
        .relu_en  (flags_w[1]),
        .acc_in   (acc_out),
        .bias_in  (bias_loaded),
        .data_out (br_out)
    );

    // ---------------- requantize ----------------
    logic [COLS*O_W-1:0] req_out;
    logic [COLS*P_W-1:0] req_mult_global;
    logic [COLS*6-1:0]   req_shift_global;
    logic [COLS*P_W-1:0] req_mult_active;
    logic [COLS*6-1:0]   req_shift_active;

    genvar q;
    generate
        for (q = 0; q < COLS; q++) begin : g_req_global
            assign req_mult_global[q*P_W +: P_W] = req_mult_w;
            assign req_shift_global[q*6 +: 6]    = req_shift_w[5:0];
        end
    endgenerate

    assign req_mult_active  = flags_w[3] ? req_mult_loaded  : req_mult_global;
    assign req_shift_active = flags_w[3] ? req_shift_loaded : req_shift_global;

    requantize #(
        .LANES (COLS),
        .P_W   (P_W),
        .O_W   (O_W)
    ) u_req (
        .req_en   (flags_w[2]),
        .mult     (req_mult_active),
        .shift    (req_shift_active),
        .acc_in   (br_out),
        .data_out (req_out)
    );

    // ---------------- ofm_writer ----------------
    // Writer fires only on the LAST tile's data_valid window. start gated
    // by data_last_tile (which lags ctrl_fsm.last_tile by the same span as
    // data_first_tile lags ctrl_fsm.first_tile, so it tracks the actual
    // accumulated-result window). Earlier tiles flow into row_accumulator
    // only — ofm_writer stays in IDLE.
    logic ow_busy_unused;
    logic ow_start_pulse;
    logic ow_data_valid;

    assign ow_start_pulse = if_start && last_tile;
    assign ow_data_valid  = data_valid && data_last_tile;

    ofm_writer #(
        .LANES  (COLS),
        .O_W    (O_W),
        .ADDR_W (ADDR_W),
        .M_W    (M_W)
    ) u_ow (
        .clk        (pclk),
        .rst_n      (presetn),
        .start      (ow_start_pulse),
        .m_count    (m_count_w[M_W-1:0]),
        .base_addr  (ofm_base_w[ADDR_W-1:0]),
        .data_in    (req_out),
        .data_valid (ow_data_valid),
        .sram_we    (ofm_sram_we),
        .sram_addr  (ofm_sram_addr),
        .sram_wdata (ofm_sram_wdata),
        .busy       (ow_busy_unused),
        .done       (ow_done)
    );

    // ---------------- intentionally unused signal taps ----------------
    // CSR registers are 32 bits per APB convention but internal ADDR_W is
    // smaller; FLAGS uses only [3:0]; REQ_SHIFT uses only [5:0]; k_count
    // upper bits beyond what k_tiles_total uses are unused.
    /* verilator lint_off UNUSEDSIGNAL */
    logic _unused_ok;
    assign _unused_ok = &{1'b0,
                          ifm_base_w[31:ADDR_W],
                          w_base_w[31:ADDR_W],
                          ofm_base_w[31:ADDR_W],
                          bias_base_w[31:ADDR_W],
                          req_mult_base_w[31:ADDR_W],
                          req_shift_base_w[31:ADDR_W],
                          flags_w[31:4],
                          req_shift_w[31:6],
                          k_count_w[31:LOG2_COLS+K_TILE_W],
                          k_count_w[LOG2_COLS-1:0],
                          m_count_w[31:M_W],
                          wl_busy_unused,
                          bl_busy_unused,
                          rp_busy_unused,
                          if_busy_unused,
                          ow_busy_unused,
                          ow_start_int,
                          tile_idx};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule
