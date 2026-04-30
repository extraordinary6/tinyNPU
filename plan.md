# tinyNPU 实施计划

用 SystemVerilog 实现一个小规模 NPU，使用 cocotb + Python 黄金模型完成功能验证。
目标是一个可跑通、可扩展、逻辑清晰的教学/实验级加速器。

---

## 1. 目标与范围

### 1.1 功能目标
- 支持 **INT8 × INT8 → INT32** 的矩阵乘法（GEMM），这是 NPU 的主计算。
- 支持 **偏置加法** 和 **ReLU 激活**，构成最小可用的线性层。
- 支持 **Conv2D via im2col**：卷积在软件侧展开成 GEMM，硬件只做矩阵乘，避免引入独立卷积数据通路。
- 支持通过 **APB** 挂到 host，CSR 配置 M/N/K 维度、基址、量化参数、启动/状态。

### 1.2 非目标（第一版不做）
- 不做完整 ISA、指令流水、分支预测。
- 不做浮点 / BF16 / FP16。
- 不做多核、DMA、片外 DDR。所有数据假设已在片内 SRAM。
- 不做 training、反向传播。
- 不做功耗/面积优化，先保证功能正确。

### 1.3 规格参数（初版）
| 项目 | 取值 | 备注 |
|------|------|------|
| PE 阵列 | 4×4 systolic | 16 个 MAC，后续可扩到 8×8 / 16×16 |
| 输入位宽 | INT8（有符号） | activation / weight 均 INT8 |
| 累加位宽 | INT32 | 防止饱和 |
| 输出位宽 | INT8（requantize）/ INT32（raw） | 通过 CSR 选择 |
| 时钟 | 单时钟域（与 APB 同源） | 暂不考虑 CDC |
| 复位 | 同步低有效 `rst_n` | 与 APB `PRESETn` 同步使用 |
| 片上 SRAM | IFM / W / OFM 三块，每块 4 KB 起步 | 大小参数化 |
| Host 总线 | APB3（简化子集） | 只支持单拍读写，无 burst |

---

## 2. 顶层架构

```
                 ┌──────────────────────────────────────────┐
                 │                tinyNPU_top               │
 host ──APB────► │  ┌──────────┐    ┌──────────────────┐    │
                 │  │ apb_csr  │───►│   ctrl_fsm       │    │
                 │  │ (regfile)│    └────────┬─────────┘    │
                 │  └──────────┘             │              │
                 │                           ▼              │
                 │  ┌────────┐  ┌──────────────────────┐    │
                 │  │ IFM SRAM├─►│                      │    │
                 │  ├────────┤  │   systolic_array     │    │
                 │  │ W   SRAM├─►│      (4x4 PE)        │    │
                 │  ├────────┤  │                      │    │
                 │  │ OFM SRAM│◄─┤  + accum + bias      │    │
                 │  └────────┘  │  + relu + requantize │    │
                 │              └──────────────────────┘    │
                 └──────────────────────────────────────────┘
```

数据流模式：**Weight-Stationary**（权重驻留 PE，激活横向流，部分和纵向累加）。
选 WS 的原因：逻辑最简、K 维度大时权重复用率高、控制逻辑最少。

---

## 3. 模块划分

### 3.1 计算通路
- `pe.sv` — 单个 MAC 单元：INT8×INT8 乘 + INT32 累加，带权重寄存器和激活 / 部分和 pipeline 寄存器。
- `systolic_array.sv` — 4×4 PE 阵列，实例化 `pe`，处理数据的横纵向传播。
- `accumulator.sv` — 阵列输出到 OFM 的累加/暂存，处理 K 维分块（K > 阵列列数时的累加）。
- `bias_relu.sv` — 加偏置 + ReLU，可通过 CSR bypass。
- `requantize.sv` — INT32 → INT8，见 §3.1.1。

#### 3.1.1 Requantize 规格（本项目采用）
采用 TFLite-lite 风格的 `multiply-shift-saturate`（对称量化，不含 zero point）：

```
product_i64 = acc_i32 * mult_i32                          // 有符号 64-bit 乘积
round_bias  = (shift == 0) ? 0 : (1'sd1 <<< (shift - 1))  // 半数向上舍入
shifted_i64 = (product_i64 + round_bias) >>> shift        // 算术右移
out_i8      = saturate(shifted_i64, [-128, +127])
```

- **`mult_i32`**：有符号 32-bit 乘数，来自 CSR `REQ_MULT`。
- **`shift`**：6-bit 无符号，来自 CSR `REQ_SHIFT`，范围 0..31（其余保留）。
- **中间位宽**：硬件内部采用 64-bit 保留位宽，消除乘法溢出。
- **舍入模式**：round-half-up（加 `1<<<(shift-1)` 再算术右移），最接近 numpy 的 `np.round` 配合 `np.right_shift` 语义。
- **饱和**：符号量值超范围直接 clip 到 `[-128, +127]`。
- **bypass**：CSR `FLAGS.REQ_EN = 0` 时直接取 `acc_i32[7:0]`（仅调试用，不保证不溢出）。
- **第一版暂不支持 per-channel**：`mult/shift` 为全局标量；per-channel 在后续阶段扩展。

黄金模型在 `tb/common/golden_model.py::requantize(acc, mult, shift)` 与 RTL bit-accurate 对齐。

### 3.2 存储与数据搬运
- `sram_wrapper.sv` — 行为级 SRAM 模型（读写各一端口），方便仿真；将来替换成实际 IP。
- `ifm_feeder.sv` — 从 IFM SRAM 读激活，按 systolic 节拍斜向注入阵列。
- `weight_loader.sv` — 把权重从 W SRAM 预载到 PE 的 weight register。
- `ofm_writer.sv` — 把激活/量化后的结果写回 OFM SRAM。

### 3.3 控制与接口
- `apb_csr.sv` — **APB3 从设备**，挂载 CSR 寄存器组。信号：`pclk / presetn / psel / penable / pwrite / paddr / pwdata / prdata / pready / pslverr`。只支持单拍读写，不支持 burst / atomic；`pready` 一直拉高；地址未映射时 `pslverr=1`。
- `csr_regfile` 内寄存器定义（初版）：

| 偏移 | 名称 | 宽度 | 读写 | 说明 |
|------|------|------|------|------|
| 0x00 | `ID` | 32 | RO | 固定魔数（如 `0x4E505500`，"NPU\0"） |
| 0x04 | `CTRL` | 32 | RW | `[0]=START`（W1S 自清零） |
| 0x08 | `STATUS` | 32 | RO | `[0]=BUSY [1]=DONE [2]=ERR` |
| 0x0C | `M` | 32 | RW | GEMM M 维 |
| 0x10 | `N` | 32 | RW | GEMM N 维 |
| 0x14 | `K` | 32 | RW | GEMM K 维 |
| 0x18 | `IFM_BASE` | 32 | RW | IFM SRAM 基址（字节） |
| 0x1C | `W_BASE` | 32 | RW | 权重 SRAM 基址 |
| 0x20 | `OFM_BASE` | 32 | RW | OFM SRAM 基址 |
| 0x24 | `BIAS_BASE` | 32 | RW | bias 基址 |
| 0x28 | `FLAGS` | 32 | RW | `[0]=BIAS_EN [1]=RELU_EN [2]=REQ_EN` |
| 0x2C | `REQ_MULT` | 32 | RW | requantize 乘数（有符号） |
| 0x30 | `REQ_SHIFT` | 32 | RW | requantize 右移位数（低 6 位有效） |

- `ctrl_fsm.sv` — 顶层 FSM：`IDLE → LOAD_W → COMPUTE → DRAIN → WRITEBACK → DONE`。
- `tinyNPU_top.sv` — 顶层集成。

---

## 4. 数据流细节

### 4.1 GEMM 映射
计算 `C[M,N] = A[M,K] × B[K,N]`（+ bias + ReLU + requantize）：
- 把 B 的一个 `4×4` tile 作为权重驻留到 PE。
- A 的 `M×4` tile 沿行方向斜向送入。
- K 维切成若干 `4` 的小段，每段权重加载一次，累加器累加。
- N 维通过外层循环换新的权重 tile。

### 4.2 时序
- PE 每拍完成一次 MAC；阵列有 `(rows + cols - 1)` 个填充/排空周期。
- `ctrl_fsm` 通过 K/M/N 计数器驱动 `ifm_feeder` 与 `weight_loader`。

---

## 5. 目录结构

```
tinyNPU/
├── plan.md                  ← 本文件
├── README.md
├── requirements.txt         ← cocotb==1.8.1, numpy<1.22
├── rtl/
│   ├── pe.sv
│   ├── systolic_array.sv
│   ├── accumulator.sv
│   ├── bias_relu.sv
│   ├── requantize.sv
│   ├── sram_wrapper.sv
│   ├── ifm_feeder.sv
│   ├── weight_loader.sv
│   ├── ofm_writer.sv
│   ├── apb_csr.sv
│   ├── ctrl_fsm.sv
│   └── tinyNPU_top.sv
├── tb/
│   ├── common/
│   │   ├── golden_model.py       # numpy 黄金模型（GEMM/ReLU/requantize）
│   │   ├── apb_driver.py         # cocotb APB master BFM
│   │   └── sram_backdoor.py      # SRAM 预加载 / 读取
│   ├── test_pe/
│   │   ├── Makefile
│   │   └── test_pe.py
│   ├── test_systolic_array/
│   ├── test_accumulator/
│   ├── test_bias_relu/
│   ├── test_requantize/
│   ├── test_apb_csr/
│   ├── test_ctrl_fsm/
│   └── test_top/
│       ├── Makefile
│       └── test_top.py           # 端到端 GEMM + bias + ReLU + requantize
├── scripts/
│   ├── run_all.sh                # 批量跑所有 cocotb 测试
│   └── lint.sh                   # verilator --lint-only
└── docs/
    └── arch.md                   # 架构细节（后续补充）
```

---

## 6. 验证策略（cocotb）

### 6.1 总体原则
- **每个 RTL 模块都有独立 cocotb testbench**，不依赖上层。
- **黄金模型用 numpy**，集中在 `tb/common/golden_model.py`；硬件输出 bit-accurate 对齐。
- **随机 + 定向用例结合**：边界值（0、极值、负数、溢出）定向；一般覆盖随机。
- **回归脚本**：`scripts/run_all.sh` 跑所有 testbench，CI 友好（后续可接 GitHub Actions）。

### 6.2 各模块验证重点
| 模块 | 关键用例 |
|------|----------|
| `pe` | 权重加载、MAC 累加、带符号乘法、累加器清零 |
| `systolic_array` | 斜向注入时序、pipeline 填充/排空、权重驻留正确性 |
| `accumulator` | K 分块累加、写回节拍 |
| `bias_relu` | ReLU 边界（0、负、正）、bypass 模式 |
| `requantize` | 正负 `mult`、`shift=0`、`shift=31`、饱和上下界、舍入半数情形 |
| `apb_csr` | 合法读写、未映射地址 `pslverr`、RO 写入忽略、W1S 自清零、复位默认值 |
| `ctrl_fsm` | 状态跳转、START/STATUS、异常 M/N/K=0 |
| `tinyNPU_top` | 端到端随机 GEMM、GEMM+bias+ReLU+requantize、最小/最大尺寸、背靠背两次计算 |

### 6.3 仿真器
- 首选 **Icarus Verilog**（免费、cocotb 支持好），不够快再切 **Verilator**。
- `make SIM=icarus` / `make SIM=verilator` 切换；Makefile 模板统一。

### 6.4 Lint
- `verilator --lint-only` 在 CI 里把 warning 当 error 挡住。

---

## 7. 里程碑 / 阶段推进

严格按顺序，**每阶段 cocotb 全绿才能进入下一阶段**。

### 阶段 0 — 环境与骨架（0.5 天）
- 建好目录结构、空文件。
- 写 `requirements.txt` 并在 py37 环境安装；跑通一个最小 cocotb demo（例如 DFF）确认 Icarus + cocotb + Python 3.7 链路通畅。

### 阶段 1 — PE 单元（1 天）
- 实现 `pe.sv`。
- `test_pe.py` 覆盖：权重加载、MAC、带符号溢出、清零。

### 阶段 2 — Systolic Array（1–2 天）
- 实现 `systolic_array.sv`，例化 4×4 PE。
- `test_systolic_array.py` 用 numpy 生成小矩阵对比结果。

### 阶段 3 — 累加 / 偏置 / 激活 / 量化(1–2 天）
- `accumulator.sv` / `bias_relu.sv` / `requantize.sv`。
- 每个都有独立 testbench；`requantize` 特别覆盖 §3.1.1 所有边界。

### 阶段 4 — 存储与 Feeder（1 天）
- `sram_wrapper.sv` 行为模型。
- `ifm_feeder.sv` / `weight_loader.sv` / `ofm_writer.sv`。
- testbench 确认时序与地址生成正确。

### 阶段 5 — APB CSR 与 FSM（1–1.5 天）
- `apb_csr.sv` + `ctrl_fsm.sv`。
- 实现 cocotb 侧的 `apb_driver.py`（BFM）并写 `test_apb_csr.py`。
- `test_ctrl_fsm.py` 覆盖所有状态跳转。

### 阶段 6 — 顶层集成（1–2 天）
- `tinyNPU_top.sv`。
- `test_top.py`：端到端 GEMM、GEMM+bias+ReLU+requantize、随机尺寸。

### 阶段 7 — 稳固与扩展（基础）
- Verilator `--lint-only -Wall` 清零（CSR 高位等 by-design unused 用 lint_off pragma 包装）。
- `scripts/lint.sh` 一键跑。
- 修整 RTL 中宽度不匹配等真实警告。

**v1 完成的功能边界**（阶段 0–7 的总产出）：
- INT8×INT8 GEMM, M×4 输入 / 4×4 weight tile / M×4 输出
- ReLU / TFLite-lite requantize（全局 mult/shift）/ INT32 raw 输出 bypass
- APB3 配置, M/N/K=0 异常处理
- v1 限制：K 必须 = 4（无 K 分块），N 必须 = 4，bias 硬接 0（无 bias loader）

---

### 阶段 8 — Per-channel requantize
**目标**：每 lane 独立的 mult/shift，是常见 INT8 推理框架的真实需求。

- 改 `requantize.sv`：`mult` 与 `shift` 端口从标量改成 LANES 个数组。
- 新 CSR：`REQ_MULT_BASE` (0x34)、`REQ_SHIFT_BASE` (0x38)；指向 W SRAM 内存放 N×INT32 mult 与 N×INT8 shift 的位置。
- 新模块 `req_param_loader.sv`：在 LOAD_W 后读两个地址，把 N×mult / N×shift 锁进内部 latch。
- 测试：扩展 `test_requantize` 覆盖 per-channel 模式；`test_top` 加一个 per-channel GEMM 用例。

预计：1–2 天。

---

### 阶段 9 — Bias loader
**目标**：让 `FLAGS.BIAS_EN` 真正生效。

- 新模块 `bias_loader.sv`（结构上类似 `weight_loader.sv`）：从 `BIAS_BASE` 读一个 4×INT32 word，锁进内部 latch。
- `ctrl_fsm` 增加 LOAD_BIAS 状态，在 LOAD_W 后并行或串行运行。
- top 把 `bias_relu.bias_in` 接 latch 输出（替换当前的硬零）。
- 测试：`test_bias_loader` 单元；`test_top` 加 `bias+ReLU+requantize` 端到端。

预计：1 天。

---

### 阶段 10 — K 分块累加
**目标**：解除 K 必须等于阵列列数（4）的限制。

- 当前架空的 `accumulator.sv` 重新接进数据通路，位于 unskew 之后、bias_relu 之前。
- `ctrl_fsm` 改造：`K_TILES = ceil(K / 4)`，每个 tile 重新 LOAD_W + COMPUTE，clr 仅在第一个 tile 拉，其它 tile en=row_valid 累加。
- `valid_gen` 在最后一个 tile 才把数据交给 ofm_writer。
- 测试：`test_top` 加 K=8 / K=12 GEMM。

预计：2 天（FSM 改造点最多）。

---

### 阶段 11 — N 分块 + 通用 unskew
**目标**：解除 N 必须等于 4 的限制。

- 把 top 内硬编码的 6 个 unskew 寄存器改为按 ROWS 参数化的 generate 实现。
- `ctrl_fsm` 增加 N tile 外循环（每 tile 跑 K_TILES 个内循环）。
- ofm_writer 写地址按 N tile 偏移。
- 测试：`test_top` 加 N=8 / N=16 GEMM。

预计：1–2 天。

---

### 阶段 12 — 8×8 阵列扩展
**目标**：把 PE 阵列从 4×4 扩到 8×8 看吞吐变化。

- `tinyNPU_top` 顶层参数 ROWS/COLS 已通用化，主要工作在工具链：通用 unskew、ifm_feeder 时序适配（stagger pipeline 深度变 7）。
- 重新跑全 testbench，验证参数化没断。
- 在 `docs/perf.md` 记录 4×4 vs 8×8 GEMM 周期数对比、PE 利用率。

预计：1–2 天。

---

### 阶段 13 — Conv2D via im2col
**目标**：纯软件 + 现有 GEMM 引擎跑 Conv2D。

- `tb/common/im2col.py`：把 `IFM[H,W,Cin]` 与 kernel `K[kh,kw,Cin,Cout]` 展开成 `A[H'W',kh*kw*Cin] @ B[kh*kw*Cin,Cout]`。
- 端到端测试 `tb/test_conv/`：Python 软件展开 → 喂 RTL → 比对 numpy `scipy.signal.correlate2d`。
- RTL 不动。

预计：1 天（Python 工作）。

---

### 阶段 14 — 性能与覆盖率
- 跑 cocotb coverage（line / toggle）和 verilator `--coverage` 收集覆盖率。
- 端到端 GEMM 周期数 vs 理论下限的对比，找 hotspot。
- 可选：把 SRAM 三块的容量、带宽参数化，跑 sweep。
- 可选：结合 GitHub Actions 把 `run_all.sh` + `lint.sh` 接入 CI。

预计：1–2 天。

---

总预计（含 v1）：**v1 8–11 工作日 + 扩展 7–10 工作日**。每阶段建议先全绿再合并到 `main`。


## 8. RTL 编码规范（强约束）

本节规则等同于编译器红线，违反即视为 bug。

### 8.1 一个 `always` 只驱动一个信号
- 每个寄存器单独一个 `always_ff`；每条组合输出单独一个 `always_comb` 或 `assign`。
- **唯一例外**：若多个信号的 clock / reset / enable / 分支结构**完全一致**，允许合并到同一个块内（此时等同于一组条件共享的寄存器）。
- 禁止在同一 `always_comb` 里用顺序 if/case 给多个彼此独立的输出赋值。

### 8.2 禁止 C / Python 风格
- 不要把临时 `logic` 当"局部变量"顺序赋值再读回来用。
- `for` 仅用于例化展开（`generate for`）和常量遍历，不作控制流。
- 不依赖赋值顺序表达依赖；每条信号的驱动路径必须在静态阅读时一目了然。
- 不把状态机写成"调用函数—返回值"形式；状态机 = `next_state` 组合逻辑 + `state` 时序逻辑，二者**分两个 always 块**。

### 8.3 类型与命名
- 统一用 `logic`；禁用 `reg` / `wire`。
- 低有效信号后缀 `_n`（如 `rst_n`），其他默认高有效。
- 模块端口顺序：`clk, rst_n, <config>, <input valid/data>, <output valid/data>`。
- 实例化一律命名连接 `.port(sig)`，禁用位置连接。

### 8.4 复位
- 默认 **同步低有效复位** (`rst_n`)，在 `always_ff @(posedge clk)` 内部首句判断：
  ```systemverilog
  always_ff @(posedge clk) begin
      if (!rst_n) q <= '0;
      else       q <= d;
  end
  ```
- 如需异步复位，必须在模块头注释说明理由。

### 8.5 字面量与宽度
- 所有字面量写明位宽与基数：`4'd0`、`8'sh7F`、`32'h0000_0000`；禁止裸 `0` / `1`。
- 有符号运算显式用 `signed`：`logic signed [7:0] a;`；混合运算前显式 `$signed` / `$unsigned`。

### 8.6 case
- 每个 `case` 必有 `default`。
- 分支互斥时用 `unique case`；有优先级时用 `priority case`；默认避免裸 `case`。

### 8.7 可综合边界
- 可综合 RTL 内禁止 `initial`、`#delay`、`$display`、`force/release`。这些只允许出现在 `tb/` 里。
- 禁用 `always @(*)`，统一用 `always_comb`；禁用 `always @(posedge clk)` 裸写，统一 `always_ff`。

### 8.8 文件约定
- 一个 `.sv` 文件一个 `module`，文件名与模块名一致。
- 头部注释只写：模块功能一句话 + 端口表；不写版权/作者。

---

## 9. 工具链

### 9.1 Python / cocotb
- **Python 解释器**：`<your-py37-env>/python.exe`（Windows 原生 Python 3.7，例如 conda 环境）。
- **cocotb**：`1.8.1`（Py3.7 最高可用版本；1.9+ 已移除 Py3.7 支持）。
- **numpy**：`<1.22`（Py3.7 最后支持的版本线）。
- **pytest**：`<=7.4.x`。
- `requirements.txt` 固定上述版本。所有 Makefile 里用 `PYTHON_BIN = <your-py37-env>/python.exe` 显式指向。

### 9.2 仿真与 lint
- **SystemVerilog 仿真器**：Icarus Verilog（主）/ Verilator（备，后期性能）。
- **Lint**：Verilator `--lint-only`，所有 warning 视为 error。
- **构建**：每个 testbench 一个 Makefile（cocotb 官方模板）。
- **Windows 注意**：Icarus + cocotb 在 Windows 原生需要 MSYS2 提供的 `make` 和 `iverilog`；`scripts/run_all.sh` 以 MSYS2 bash 为目标环境。

### 9.3 编辑器
- 推荐 svlangserver + VSCode。

---

## 10. 风险与待定事项

| 项 | 描述 | 处置 |
|----|------|------|
| R1 | Icarus 对 SV 某些构造（`interface`、高级 `always_ff`）支持有限 | RTL 写法保守；首版不用 `interface`，必要时切 Verilator |
| R2 | Windows 原生运行 cocotb 需要 MSYS2 工具链，路径/行尾易出坑 | 阶段 0 优先跑通 DFF demo，确认工具链后再推进；文档给出 MSYS2 安装步骤 |
| R3 | ~~requantize 算子语义未定~~ | **已决**：TFLite-lite mult-shift-saturate，详见 §3.1.1 |
| R4 | ~~顶层 handshake 协议未定~~ | **已决**：APB3 简化子集，详见 §3.3 |
| R5 | 是否需要 DMA / 外部内存 | 第一版不做，顶层接口预留 |
| R6 | Py3.7 + cocotb 1.8.1 老版本可能缺某些新 API | 文档/脚本一律按 1.8.1 写；遇到新 API 缺失直接回退 |

---

## 11. 下一步（阶段 0 具体动作）

1. 按 §5 建目录与空文件骨架。
2. 写 `requirements.txt`：
   ```
   cocotb==1.8.1
   numpy<1.22
   pytest<7.5
   ```
3. 在 Python 3.7 环境（如 `conda activate py37`）下 `pip install -r requirements.txt`。
4. 确认 MSYS2 环境安装 `iverilog` 与 `make`；`scripts/run_all.sh` 给出调用样例。
5. 写一个最小 `dff.sv` + `test_dff.py` + `Makefile`（`PYTHON_BIN` 指向 py37），跑通整个 cocotb 链路。
6. 全绿后进入阶段 1，开始写 `pe.sv`，严格遵守 §8 编码规范。
