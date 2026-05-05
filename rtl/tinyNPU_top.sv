// tinyNPU_top.sv
// Top-level integration. APB3-controlled GEMM with optional bias / ReLU /
// (per-channel) requantize, K-tile accumulation across multiple weight
// tiles, and N-tile sweep across multiple column blocks.
//
// Datapath: ifm_feeder -> systolic_array -> unskew -> row_accumulator
//           -> bias_relu -> requantize -> ofm_writer.
// Control:  apb_csr (CSR file) drives ctrl_fsm. ctrl_fsm runs an outer
//           N-tile loop and an inner K-tile loop. For each N tile it loops
//           LOAD_W <-> COMPUTE k_tiles_total times accumulating partial
//           sums; on the first K tile of every N tile it optionally runs
//           LOAD_BIAS (if bias_en) and LOAD_REQ (if pch_req_en); on the
//           last K tile it runs WRITEBACK.
//
// SRAM layout (caller responsibility):
//   IFM SRAM: A[M, K] tile-major over K. Tile k slice A[:, k*COLS:(k+1)*COLS]
//             at addresses ifm_base + k*M ... + (k+1)*M - 1. Reused across
//             every N tile.
//   W   SRAM: weight tile (n_tile, k_tile) — slab
//             B[k*COLS:(k+1)*COLS, n*COLS:(n+1)*COLS] — at address
//             w_base + n*K_TILES + k. Outer N, inner K (matches the
//             sequential w_base_tile counter).
//   BIAS SRAM: one COLS x INT32 word per N tile, at bias_base + n_tile_idx.
//   REQ SRAM (in W SRAM): one mult word + one shift word per N tile, at
//             req_mult_base + n_tile_idx and req_shift_base + n_tile_idx.
//   OFM SRAM: tile-major over N. Tile n's M output rows at addresses
//             ofm_base + n*M ... + (n+1)*M - 1.
//
// Pipeline latency from each tile's if_start to first OFM write = 9
// (FSM startup + SRAM read + 4-deep PE column + 3-deep unskew). The
// ofm_writer fires only on the last K tile of each N tile.

module tinyNPU_top #(
    parameter int ROWS     = 4,
    parameter int COLS     = 4,
    parameter int A_W      = 8,
    parameter int O_W      = 8,
    parameter int P_W      = 32,
    parameter int ADDR_W   = 12,
    parameter int M_W      = 16,
    parameter int M_MAX    = 64,
    parameter int K_TILE_W = 8,
    parameter int N_TILE_W = 8
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
    logic                    compute_done;
    logic [K_TILE_W-1:0]     tile_idx;
    logic                    first_tile, last_tile;
    logic [N_TILE_W-1:0]     n_tile_idx;
    logic                    n_first_tile, n_last_tile;
    logic [K_TILE_W-1:0]     k_tiles_total;
    logic [N_TILE_W-1:0]     n_tiles_total;

    // K must be a multiple of COLS, N must be a multiple of COLS; software
    // is responsible. k_tiles_total = k_count >> $clog2(COLS),
    // n_tiles_total = n_count >> $clog2(COLS).
    localparam int LOG2_COLS = $clog2(COLS);
    assign k_tiles_total = k_count_w[LOG2_COLS +: K_TILE_W];
    assign n_tiles_total = n_count_w[LOG2_COLS +: N_TILE_W];

    ctrl_fsm #(
        .K_TILE_W (K_TILE_W),
        .N_TILE_W (N_TILE_W)
    ) u_fsm (
        .clk           (pclk),
        .rst_n         (presetn),
        .start         (start_pulse),
        .m_count       (m_count_w),
        .n_count       (n_count_w),
        .k_count       (k_count_w),
        .k_tiles_total (k_tiles_total),
        .n_tiles_total (n_tiles_total),
        .bias_en       (flags_w[0]),
        .pch_req_en    (flags_w[3]),
        .wl_start      (wl_start),
        .wl_done       (wl_done),
        .bl_start      (bl_start),
        .bl_done       (bl_done),
        .rp_start      (rp_start),
        .rp_done       (rp_done),
        .if_start      (if_start),
        .compute_done  (compute_done),
        .ow_start      (ow_start_int),
        .ow_done       (ow_done),
        .tile_idx      (tile_idx),
        .first_tile    (first_tile),
        .last_tile     (last_tile),
        .n_tile_idx    (n_tile_idx),
        .n_first_tile  (n_first_tile),
        .n_last_tile   (n_last_tile),
        .busy          (busy),
        .done          (done),
        .err           (err)
    );

    // ---------------- per-tile address computation ----------------
    // weight_loader: every (n_tile, k_tile) pair occupies one packed word.
    // Counter advances once per wl_done, so the addresses are issued in
    // outer-N inner-K order: w_base, w_base+1, ... w_base + (N_TILES*K_TILES - 1).
    logic [ADDR_W-1:0] w_base_tile;
    always_ff @(posedge pclk) begin
        if (!presetn)         w_base_tile <= {ADDR_W{1'b0}};
        else if (start_pulse) w_base_tile <= w_base_w[ADDR_W-1:0];
        else if (wl_done)     w_base_tile <= w_base_tile + {{(ADDR_W-1){1'b0}}, 1'b1};
    end

    // ifm_feeder: each K tile occupies M consecutive addresses. Track an
    // offset register that resets at the start of a kick AND at every
    // N-tile boundary (since A is reused across N tiles), and advances by
    // m_count when entering a non-first K tile's COMPUTE.
    logic [ADDR_W-1:0] ifm_base_tile;
    always_ff @(posedge pclk) begin
        if (!presetn)                              ifm_base_tile <= {ADDR_W{1'b0}};
        else if (start_pulse)                       ifm_base_tile <= ifm_base_w[ADDR_W-1:0];
        else if (ow_done && !n_last_tile)           ifm_base_tile <= ifm_base_w[ADDR_W-1:0];
        else if (if_start && !first_tile)           ifm_base_tile <= ifm_base_tile + m_count_w[ADDR_W-1:0];
    end

    // bias_loader / req_param_loader: one address per N tile. Use formula
    // (n_tile_idx changes only at WRITEBACK->LOAD_W edges, well before the
    // loaders sample base_addr in their FETCH state).
    logic [ADDR_W-1:0] bias_base_tile;
    logic [ADDR_W-1:0] req_mult_base_tile;
    logic [ADDR_W-1:0] req_shift_base_tile;
    assign bias_base_tile      = bias_base_w[ADDR_W-1:0]      + {{(ADDR_W-N_TILE_W){1'b0}}, n_tile_idx};
    assign req_mult_base_tile  = req_mult_base_w[ADDR_W-1:0]  + {{(ADDR_W-N_TILE_W){1'b0}}, n_tile_idx};
    assign req_shift_base_tile = req_shift_base_w[ADDR_W-1:0] + {{(ADDR_W-N_TILE_W){1'b0}}, n_tile_idx};

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
        .mult_base_addr  (req_mult_base_tile),
        .shift_base_addr (req_shift_base_tile),
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
        .base_addr  (bias_base_tile),
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

    // ---------------- unskew (parameterized over COLS) ----------------
    // Lane c (c < COLS-1) is delayed by (COLS-1-c) cycles; lane COLS-1 is
    // a passthrough. Implemented as one shift register per lane, packed
    // contiguously into a single 1D unpacked array sk to keep all storage
    // cells driven and read.
    localparam int N_STAGES = (COLS - 1) * COLS / 2;
    logic [P_W-1:0] sk [0:N_STAGES-1];
    logic [COLS*P_W-1:0] psum_unskewed;

    genvar uc, us;
    generate
        for (uc = 0; uc < COLS - 1; uc++) begin : g_unskew_lane
            localparam int OFFSET = uc * (COLS - 1) - uc * (uc - 1) / 2;
            localparam int STAGES = COLS - 1 - uc;

            always_ff @(posedge pclk) begin
                if (!presetn) sk[OFFSET] <= {P_W{1'b0}};
                else          sk[OFFSET] <= sys_psum[uc*P_W +: P_W];
            end
            for (us = 1; us < STAGES; us++) begin : g_unskew_stage
                always_ff @(posedge pclk) begin
                    if (!presetn) sk[OFFSET + us] <= {P_W{1'b0}};
                    else          sk[OFFSET + us] <= sk[OFFSET + us - 1];
                end
            end
            assign psum_unskewed[uc*P_W +: P_W] = sk[OFFSET + STAGES - 1];
        end
    endgenerate
    assign psum_unskewed[(COLS-1)*P_W +: P_W] = sys_psum[(COLS-1)*P_W +: P_W];

    // ---------------- valid_gen FSM (per-tile data_valid window) ----------------
    // Pipeline latency from if_start to first data_valid:
    //   1 (FSM startup) + 1 (SRAM read) + ROWS (PE column depth)
    //   + (COLS-1) (deepest unskew lane) = ROWS + COLS + 1.
    // The counter spans cycles in VG_LAT (one cycle per increment) and
    // transitions to VG_VAL when it reaches LATENCY - 2 = ROWS + COLS - 1.
    localparam int VG_LAT_TARGET = ROWS + COLS - 1;
    localparam int VG_LAT_W      = (VG_LAT_TARGET == 0) ? 1 : $clog2(VG_LAT_TARGET + 1);

    typedef enum logic [1:0] {
        VG_IDLE = 2'd0,
        VG_LAT  = 2'd1,
        VG_VAL  = 2'd2
    } vg_state_t;

    vg_state_t vg_state, vg_state_n;
    logic [VG_LAT_W-1:0] vg_lat_cnt;
    logic [M_W-1:0]      vg_val_cnt;

    always_ff @(posedge pclk) begin
        if (!presetn) vg_state <= VG_IDLE;
        else          vg_state <= vg_state_n;
    end

    always_comb begin
        vg_state_n = vg_state;
        unique case (vg_state)
            VG_IDLE: if (if_start) vg_state_n = VG_LAT;
            VG_LAT:  if (vg_lat_cnt == VG_LAT_W'(VG_LAT_TARGET)) vg_state_n = VG_VAL;
            VG_VAL:  if (vg_val_cnt == m_count_w[M_W-1:0] - {{(M_W-1){1'b0}}, 1'b1})
                        vg_state_n = VG_IDLE;
            default: vg_state_n = VG_IDLE;
        endcase
    end

    always_ff @(posedge pclk) begin
        if (!presetn)                vg_lat_cnt <= {VG_LAT_W{1'b0}};
        else if (vg_state != VG_LAT) vg_lat_cnt <= {VG_LAT_W{1'b0}};
        else                         vg_lat_cnt <= vg_lat_cnt + {{(VG_LAT_W-1){1'b0}}, 1'b1};
    end

    always_ff @(posedge pclk) begin
        if (!presetn)                vg_val_cnt <= {M_W{1'b0}};
        else if (vg_state != VG_VAL) vg_val_cnt <= {M_W{1'b0}};
        else                         vg_val_cnt <= vg_val_cnt + {{(M_W-1){1'b0}}, 1'b1};
    end

    logic data_valid;
    assign data_valid = (vg_state == VG_VAL);

    // ---------------- data-side tile counters ----------------
    // ctrl_fsm.first_tile / last_tile flip on if_done edges (control side),
    // but the corresponding data window (after the unskew tail) extends a
    // few cycles past if_done. Track tile boundaries on the valid_gen side
    // so row_accumulator and ofm_writer see consistent first/last edges
    // across the M-row data window of the relevant tile.
    //
    // data_tile_idx wraps back to 0 at every k_tiles_total-th data window
    // (i.e., at every N tile boundary). data_n_tile_idx increments at the
    // same edge.
    logic data_done_pulse;
    assign data_done_pulse = (vg_state == VG_VAL) && (vg_state_n == VG_IDLE);

    // ctrl_fsm waits for the data-side drain rather than ifm_feeder's own
    // done pulse: the latter fires after the input stream finishes but
    // before the systolic array drains, so using it would let LOAD_W
    // overwrite PE weights mid-flight (especially as ROWS+COLS grow).
    assign compute_done = data_done_pulse;

    logic [K_TILE_W-1:0] data_tile_idx;
    logic [N_TILE_W-1:0] data_n_tile_idx;
    logic                data_n_advance;

    assign data_n_advance = data_done_pulse
                            && (data_tile_idx == k_tiles_total - {{(K_TILE_W-1){1'b0}}, 1'b1});

    always_ff @(posedge pclk) begin
        if (!presetn)             data_tile_idx <= {K_TILE_W{1'b0}};
        else if (start_pulse)     data_tile_idx <= {K_TILE_W{1'b0}};
        else if (data_n_advance)  data_tile_idx <= {K_TILE_W{1'b0}};
        else if (data_done_pulse) data_tile_idx <= data_tile_idx + {{(K_TILE_W-1){1'b0}}, 1'b1};
    end

    always_ff @(posedge pclk) begin
        if (!presetn)            data_n_tile_idx <= {N_TILE_W{1'b0}};
        else if (start_pulse)    data_n_tile_idx <= {N_TILE_W{1'b0}};
        else if (data_n_advance) data_n_tile_idx <= data_n_tile_idx + {{(N_TILE_W-1){1'b0}}, 1'b1};
    end

    logic data_first_tile;
    logic data_last_tile;
    assign data_first_tile = (data_tile_idx == {K_TILE_W{1'b0}});
    assign data_last_tile  = (data_tile_idx == k_tiles_total - {{(K_TILE_W-1){1'b0}}, 1'b1});

    // ofm_base_tile: advances by m_count at every data-side N-tile boundary.
    logic [ADDR_W-1:0] ofm_base_tile;
    always_ff @(posedge pclk) begin
        if (!presetn)            ofm_base_tile <= {ADDR_W{1'b0}};
        else if (start_pulse)    ofm_base_tile <= ofm_base_w[ADDR_W-1:0];
        else if (data_n_advance) ofm_base_tile <= ofm_base_tile + m_count_w[ADDR_W-1:0];
    end

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
    // Fires on the LAST K tile of every N tile. start gated by ctrl_fsm's
    // last_tile (control-side); data_valid gated by data_last_tile (data-side
    // counter, lags last_tile by the unskew tail). Earlier K tiles flow
    // into row_accumulator only.
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
        .base_addr  (ofm_base_tile),
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
    // smaller; FLAGS uses only [3:0]; REQ_SHIFT uses only [5:0]; k_count and
    // n_count upper bits beyond what {k,n}_tiles_total use are unused; the
    // ctrl_fsm exposes tile_idx / n_first_tile that aren't consumed in top.
    /* verilator lint_off UNUSED */
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
                          n_count_w[31:LOG2_COLS+N_TILE_W],
                          n_count_w[LOG2_COLS-1:0],
                          m_count_w[31:M_W],
                          wl_busy_unused,
                          bl_busy_unused,
                          rp_busy_unused,
                          if_busy_unused,
                          if_done,
                          ow_busy_unused,
                          ow_start_int,
                          tile_idx,
                          n_first_tile};
    /* verilator lint_on UNUSED */

endmodule
