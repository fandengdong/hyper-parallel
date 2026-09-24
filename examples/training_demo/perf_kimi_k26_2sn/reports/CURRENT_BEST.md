# 当前最优配置与成绩单（2026-09-18 夜，均衡档口径）

> 全部数字都在 **`fix_router: true`（强制负载均衡）** 口径下、**同窗口交错两轮**取得，
> 窗口分辨力约 **0.1%**。详细过程与机制见 `overnight_optimization_20260916.md`。

## 冠军配置

| 规模 | 配置文件 | 最小步时 | padded tok/s | 峰值 allocated |
|---|---|---|---|---|
| **512 卡（4 超节点）** | `my_workspace/4snr6/pf3_dp128.yaml` | **18.2617 s** | **228,873** | 39.19 GB |
| **384 卡（3 超节点）** | `my_workspace/3sn/pf3_3sn_dp128_edp3.yaml` | **18.1459 s** | 166,342 | 39.19 GB |
| **256 卡（2 超节点）** | `my_workspace/fam256loc/gbs4096_rbwd_true.yaml`（GBS 4096 = 16×micro-batch，dp128/ep128 + swap×128） | **293.8240 s**（= **71.73 ms/样本**，**等效 GBS-256 = 18.364 s**） | **113,400** | 49.10 GB |
| 256 卡阶梯（同窗口，供对照） | 512 / 1024 / 2048 → `gbs512_rbwd_true.yaml` / `gbs1024.yaml` / `gbs2048_rbwd_true.yaml` | 38.5778 / 75.0517 / 148.5950 s（= 75.35 / 73.29 / 72.56 ms/样本；等效 19.29 / 18.76 / 18.57 s） | 108,723 / 112,088 / 112,900 | 48.87 / 48.91 / 48.97 GB |
| 256 卡（旧冠军基线） | `my_workspace/fam256loc/pf3_oswap_st128.yaml`（GBS 256 = 1 micro） | 20.4957 s（= 80.06 ms/样本） | 102,322 | 41.49 GB |
| **128 卡（1 超节点）** | `my_workspace/fam128/ep64_oswap_st128.yaml`（ep64/edp2 + swap×128） | **21.38 s** | ~49,000 | 45.88 GB |

> **2026-09-20 Round-1/2：2SN 冠军走完 GBS 阶梯（256 → 4096，等效步时 20.50 → 18.364 s）**
>
> `step = 71.524 ms × GBS + 2.017 s`（4 点最小二乘，偏差 ≤0.83%），**渐近线 18.310 s**：
>
> | GBS | micro | min 步时 | ms/样本 | 等效 256 | padded tok/s | MFU | 峰值 |
> |---|---|---|---|---|---|---|---|
> | 256（旧冠军） | 1 | 20.4957 s | 80.06 | 20.50 | 102,322 | 25.13% | 41.49 GB |
> | 512 | 2 | 38.5778 s | 75.35 | 19.29 | 108,723 | 26.71% | 48.87 GB |
> | 1024 | 4 | 75.0517 s | 73.29 | 18.76 | 112,088 | 27.53% | 48.91 GB |
> | 2048 | 8 | 148.5950 s | 72.56 | 18.574 | 112,900 | 27.73% | 48.97 GB |
> | **4096** | **16** | **293.8240 s** | **71.73** | **18.364** | **113,400** | **28.05%** | **49.10 GB** |
>
> 相对旧冠军（GBS 256）**每样本 −10.4%**，MFU 25.13% → 28.05%。收益**全部来自摊薄每步固定开销 `b`**
> （每样本成本 `a` 不随 GBS 变 ⇒ **加大 GBS 没有隐藏任何通信**）；**峰值从 2 micro 起就持平**
> （寄存机制，见 `reshard_after_backward_landmine_20260920.md` §5.2/§5.3）。
> **4096 距渐近线只剩 0.3% ⇒ GBS 杠杆已吃干**，后续优化必须落在 `a ≈ 71.7 ms/样本` 上
> （当前 device 账本与"还剩什么"见 `opt_loop_2sn_20260920.md`）。
> 同批判决：`reshard_after_backward: false`（省 3/8 的 all-gather）**差 0.33 GB 装不下，且收益上限
> ~1.4%，已封档**（同上 §5.3.2）。

> **2026-09-20：2SN 换冠军（同窗口交错两轮 C,T,C,T，同一 32 节点）**
> GBS 256 vs GBS 512，其余配置完全相同，**两臂均 `reshard_after_backward: true`**：
>
> | 臂 | GBS | min 步时 | 中位步时 | ms/样本 | padded tok/s | MFU@min | 峰值 |
> |---|---|---|---|---|---|---|---|
> | C1 / C2（旧冠军） | 256 | 20.5102 / 20.6290 s | 20.5994 / 20.6958 s | 80.12 / 80.58 | 102,249 / 101,660 | 25.12 / 24.97% | 41.49 GB |
> | T1 / T2（新） | 512 | **38.8095 / 38.5778 s** | 38.8486 / 38.7389 s | **75.80 / 75.35** | 108,074 / **108,723** | 26.55 / **26.71%** | 48.87 GB |
>
> **每样本 −5.95%（min 口径）/ −6.07%（中位口径），padded tok/s +6.33%，MFU 24.97→26.71%，代价 +7.39 GB**
> （48.87 GB，距 ~61.3 GB 上限仍余 ~12 GB）。两轮方向一致（round1 −5.39%、round2 −6.50%）；
> 控制臂 min 20.5102 / 20.6290 s 落在历史 band 20.49–20.69 内 ⇒ **窗口洁净，结论成立**。
> 复现：`bash my_workspace/run_gbs_ab_interleaved.sh` ＋ `python3 my_workspace/analyze_gbs_ab.py`。
>
> ⚠️ **该配置原先不成立**：`gbs512.yaml` 以 `reshard_after_backward: false` 跑会在 step 0 OOM
> （累积=2 ⇒ 全量参数常驻）。修正该 flag 后才有这个成绩，详见
> `reshard_after_backward_landmine_20260920.md`。

**规模效应全景（min 步时 / 每卡 padded tok/s，均 `fix_router` 均衡档）**：

| 规模 | 最优配置 | 步时 | 每卡 tok/s | 关键点 |
|---|---|---|---|---|
| 512 卡（4SN） | `4snr6/pf3_dp128.yaml`（无 swap） | 18.26 | 447 | dp_shard 本地化 −6.9% |
| 384 卡（3SN） | `3sn/pf3_3sn_dp128_edp3.yaml`（无 swap） | 18.15 | 433 | 同上 |
| 256 卡（2SN） | `fam256loc/pf3_oswap_st128.yaml` | **20.49** | **398** | swap 刚需；**swap_times 曲线 16→64→128→256：21.02→20.70→20.58→20.74**，st128 最优 |
| 128 卡（1SN） | `fam128/ep64_oswap_st128.yaml` | **21.38** | **383** | edp=1，m+v 必须 swap；st128 比 st16 −2.7% |

**等长 a2a（`HP_EP_EQUAL_A2A`，提交 `b4873e4a`，default-off）**：2SN 3 轮交错 A/B 无收益
（mode0 min 20.99–21.04 / mode1 20.76–21.06 s，噪声内）——与 512 卡 +1.24% 同属边际，维持默认关。

**2 超节点专项结论（2026-09-18）**：ep 钉 128（=1 超节点）时，1T 专家参数的分片维 = 卡数/128，
2 超节点只有 edp=2 ⇒ 每卡参数+梯度+优化器态是 4 超节点（edp=4）的 **2 倍**，峰值 56.6 GB、
余量仅 2.3 GB ⇒ prefetch 被饿死，步时 ~38 s 且剧烈抖动（f256/g256 同模式，非配置错误）。
**开优化器态 swap（省 ~32 GB）后 21.06 s 且稳定** —— 2SN 的 swap 是刚需而非 +7.3% 的亏本买卖。
对照：`r2sn_pf3_dp128`（无 swap）24.9–41.0 s / 56.59 GB；`r2sn_pf3_dp128_oswap` 21.06–22.84 s / 41.49 GB。

**2SN + swap 配置扫描（单轮，min 步时；彼此之间 ~1% 噪声内）**：

| 臂 | min 步时 | 峰值 GB | 备注 |
|---|---|---|---|
| `fam256loc/ep64_oswap.yaml` | **20.92** | **36.29** | a2a 缩到半超节点；省 5 GB 显存 |
| `fam256loc/chunk8192_oswap.yaml` | 21.03 | 46.44 | loss chunk 翻倍吃 +5 GB，无时间收益 |
| `fam256loc/pf3_oswap.yaml`（dp128） | 21.06 | 41.49 | 参照 |
| `fam256/base_oswap.yaml`（dp256） | 21.20 | 40.78 | swap 解除内存压力后，dp_shard 本地化收益消失（21.20 vs 21.06 ≈ 噪声） |

**结构性上限**：ep 钉 128 时 2SN 每卡专家权重/梯度是 4SN 的 2 倍（FSDP 通信量同理），
plus swap 暴露成本 —— 与 4SN 冠军 18.26 s 的差距大部属结构性。
已判：等长 a2a 无收益（噪声内）；**swap_times 16→64→128 逐步 −1.5%/−0.6%，256 反弹**，st128 封项。

关键 delta（相对 `4sn/pf3.yaml` / `3sn/pf3_3sn.yaml`，即 `dp_shard = 卡数`）：

```yaml
fsdp_config:
  dp_shard_size: 128        # 512 卡：改自 512（复制维 4）；384 卡：改自 384（复制维 3）
```

**机制**：稠密 FSDP 网格是 `(fsdp_replicate, fsdp_shard, tp)`（shard 在内、行主序）⇒
`dp_shard = 卡数` 时分片组跨 3–4 个超节点；`128` 时正好落在**一个超节点内**，
于是 all-gather / reduce-scatter 从跨超变本地。代价是复制维上一份跨超的**梯度** all-reduce。

## 今晚累计收益（512 卡）

| 阶段 | 最小步时 | padded tok/s |
|---|---|---|
| 起点（两个修复前） | 21.29 – 21.32 s | 196.8k |
| + 两个修复（上一轮，已提交） | 19.58 / 19.62 s | 213.7k |
| + `dp_shard` 512 → 128 | **18.2617 s** | **228.9k** |
| **合计** | **−14.3%** | **+16%** |

> 夜间（Round-23→35）的完整测试过程与数字见 `NIGHTLY_SUMMARY_20260920.md`。

> **2026-09-21 复现确认（`r36_champ100`，100 步请求 / 实际 57 步，数据耗尽即停）**：
> min **293.971 s**、median 316.282 s、max 331.363 s、mean 313.731 s，peak **49.099 GB**，
> MFU **0.2804**、HFU **0.3739**（=MFU×4/3，`activation_checkpoint.mode: full`），
> **71.770 ms/样本 ⇒ 等效 GBS-256 = 18.3732 s** —— 与记录的 18.364 s 相差 **0.05%**，冠军可复现。
> 注意：**均衡 router（`fix_router: true`）下 max/min 仍达 1.127×**，说明抖动来自集群/调度而非负载不均。
> 真实路由的 100 步对照见 `r37_realrouter100`（配置 `fam256loc/gbs4096_realrouter.yaml`）。

## 已封档的方向（都有可复现实测，不要再重试）

| 方向 | 结论 | 依据 |
|---|---|---|
| `dp_shard` 本地化 | ✅ **−6.93%（512）/ −4.13%（384）** | 两轮交错，曲线 64/96/128/192/384 已扫 |
| `edp_shard↓`（专家复制） | ❌ +23%（edp1）/ +15.7%（edp2）；无 swap 则内存抖动 | 通信族 profile：AG −45% 但 **AR ×106**、idle +3.5 s |
| seq 形状 | ❌ 4096 每 token +4.8%；16384 −25.9% | 多轮 + 384 卡实测 |
| MBS | ❌ 512 卡只能 1；MBS2 每 token −4.7% | `dp_shard = GBS` 锁死 |
| FSDP `enable_offload` | ❌ −9.94 GB 但 +18%，且**不是**优化器卸载 | 代码 + 实测对账 |
| 优化器态 swap | ⚠️ **512 卡档**：−9.95 GB / +7.3%；**2SN 档：−15.1 GB，且缺它时 24.9–41.0 s 抖动**（vs 有 swap 21.06–22.84 s）⇒ 2SN 上是**刚需**，不是可交易的储备 | 提交 `2d7e14e9` |
| AC（重计算） | ❌ 关掉装不下；且 selective ≈ full | 6 次 OOM + 512 卡对照；**2SN 复核（Round-23）**：`HP_AC_SELECTIVE_LAYERS=120` 的峰值与全 AC **逐位相同（41.4853 GB）**、步时在噪声内；`N=200` **OOM @58.94 GiB**（天花板）⇒ **不存在既装得下又有收益的 N** |
| a2a 调用粒度 | ❌ 切块 +11.4%/+39.7%；融合 +4.9% | 两轮交错 |
| 等长 a2a 内核 | ❌ 冠军 +1.24%（惰性等待更值钱）；**但非重叠路径 −15.6%** | 2×2，提交 `f9d27d24` |
| CP=2 | ❌ 8K 差 450 MiB、4K 差 0.2 GB（缓冲区尺寸固定） | OOM 签名逐条解析 |
| TP=2 | ❌ 384 卡 +141%~167%；512 卡直接 OOM | 先开发了 TP-capable loss（`eabc64f3`） |
| 保序 IP 列表 | ⚠️ dp128 −0.14%（噪声）；旧 dp512 −0.39% | 与 `dp_shard` 本地化同源，不叠加 |
| **`torch.compile`（NPU 后端）** | ❌ **已真正测完：融合确实发生，但 min 步时 +0.78%、首步 +26.5 分钟编译** | 前四轮（r22/r23）的"中性"是**假象**：`INCLUDE=attn` 命中的 108 个块正是 `vision_tower.layers.N.attn.*` 的 108 条规格（冻结、<1% 步时、单 matmul 无可融合）⇒ 内核清单当然不变。修好选择器（`HP_COMPILE_DESCENDANTS=1` + `language_model.*self_attn$` ⇒ **61 个文本注意力**）并绕过四道工具链墙（节点本地缓存 / `detect_flattened_dims` shim / `force_fallback_kernel_id` / `HP_COMPILE_ROPE_EAGER=1`）后，r30 产出 **7088 个真融合 kernel**（`fused_cat`/`ones_triu`/`clone_transpose`/rotary 前向）；`cfgdiff` 确认与对照只差 `compile.enabled`：**min 20.6198 → 20.7797 s（+0.78%）**，peak 均 41.4853 GB。⇒ 重算子已是厂商手调 `aclnn*`，编译器只能融访存受限的小算子，打不过 CANN 向量算子。**不要再在 compile 上加注** |
| **a2a 层次化重构（DeepEP 式）** | ❌ **位精确但慢 29.9%** | `r24`（16 节点 / 128 rank = 真实 ep_size，7.34 MB/peer）：flat 生产原语 **42.767 ms / 22.0 GB/s** vs 两相层次化 **61.017 ms / 15.4 GB/s**，`bitwise identical: True`。均匀计划下每 rank 跨节点字节数不变，层次化只是换切分方式（intra + inter + 两次本地重排 ≈ 4× 搬运）⇒ **"a2a 有 3× 余量"是跨模式类比、不可兑现；a2a 的问题是没被 overlap 藏住（隔离 22.0 vs 步内 14.0 GB/s），不是算法或带宽** |

## a2a 两项代码工作：已做，且 2SN 结论已出

1. **sync 路径的列表形式 a2a（已实现）**：`_EPAllToAllUneven.forward` 增加 env 门控
   `HP_EP_A2A_SINGLE=1`（默认关、数值等价），把列表形式 `dist.all_to_all` 换成
   `all_to_all_single` + splits（微基准 52.9 → 106.4 GB/s）。
   **范围**：只影响这个**同步**入口；**当前冠军走 async 路径**（`ep_all_to_all_async` →
   `_AsyncA2ALazyBwd`，其 forward 本来就是 `torch.distributed._functional_collectives.all_to_all_single`），
   所以对冠军中性 —— 这一点与本文档原判断一致（"对当前冠军中性（它走 async 路径）"）。
2. **惰性 + 无 split 的 a2a（已实现并已判定）**：`HP_EP_EQUAL_A2A=1` 的惰性分支
   `_EPAllToAllEqualLazy`（旁路流 + event 句柄）已落地。**2SN 实测为中性**：
   `r3_a2aeq` min 74.972 s vs 同窗口对照 `r3_ctrl` 74.902 s（**+0.09%**，GBS 1024）。
   机理：融合 dispatch 未启用时该惰性路径确实会被走到，但暴露的 a2a 是**结构性依赖**
   （dispatch → 专家 GEMM 硬依赖，shared-expert 重叠窗口实测只有 0.23 s），
   没有独立工作可用来填这 3.09 s ⇒ **原"−3%"预期在 2SN 不成立**。
   完整三重封档见 `opt_loop_2sn_20260920.md` §22.3。

## 群集与工具

- 节点集：`my_workspace/ip_all.txt`（64 台 = 4 超节点）、`my_workspace/ip_sn134.txt`（48 台 = 3 超节点，避开被同事占用的 SN2）。
- 跑臂：`bash my_workspace/run_arm_guarded.sh <tag> <yaml>`（双重闸门：节点全空 + 每台能看到仓库）；
  A/B：`run_file_ab.sh`（同 yaml 换配置）、`run_sx_ab.sh`（同 yaml 换 env）、`run_clean_arm.sh`（臂前后自动清场，**OOM 臂必用**）。
- 清场：`my_workspace/clean_all.sh`（含 **rendezvous 端口回收**；OOM 臂会把端口占住，导致下一臂 `EADDRINUSE` 而 62 台空等）。
- 分析：`prof_ledger.py`（设备账本）、`compare_ranks.py`（逐 rank 对照）、`probe_h2d_bw.py`（H2D/D2H 带宽）、`bench_a2a.py`（a2a 微基准）。
- 提交：`2d7e14e9`（优化器态 swap）、`f9d27d24`（等长 a2a）、`eabc64f3`（TP-capable loss）、`f353960b`（配置契约测试同步）。

## 真实数据 8K packing 训练（r51，2026-09-22 起）

吞吐冠军（上表）是固定长度合成数据的口径。真实数据侧另有一条**在跑**的线：
`coco2017_2sn_8k_packed_cap45.yaml`（COCO2017 LLaVA-instruct，每窗拼满 8K，10 样本/10 图每窗，
填充率 93.4%）。稳态 50–58 s/step、peak 50.37 GB、real tok/s 8,210、supervised/padded 22.6%
（padded 臂 2.3%），等步数 loss 下降是 padded 臂的 **1.89×**。
细节 / OOM 账本 / 口径警告见 `my_workspace/COCO8K_PACKED_RUN.md`。
