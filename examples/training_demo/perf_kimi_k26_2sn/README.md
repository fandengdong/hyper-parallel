# Kimi-K2.6 VLM 性能冠军复现包

把当前各规模的**最优配置**与**成绩单**放进仓库：checkout 本 PR 后，按 §5 就能拉起同样的训练。

**口径**：所有数字都在 `fix_router: true`（强制负载均衡）下、**同窗口交错多轮**取得，窗口分辨力约 0.1%；
模型是 K2.6 61 层 VLM（文本塔 61 层 × 384 专家 + 视觉塔 27 层），`seq 8192`；
跨 GBS 比较一律归一到 **等效 GBS-256 步时**（`min步时 × 256 / GBS`）。

---

## 1. 实测结果

### 1.1 各规模冠军

| 规模 | 配置文件 | min 步时 | padded tok/s | 峰值 allocated |
|---|---|---|---|---|
| **256 卡（2SN）** | `configs/256card_2sn/gbs4096_rbwd_true.yaml` | **293.8240 s**（71.73 ms/样本；等效 GBS-256 **18.364 s**） | **113,400** | **49.10 GB** |
| 512 卡（4SN） | `configs/512card_4sn/pf3_dp128.yaml` | **18.2617 s** | 228,873 | 39.19 GB |
| 384 卡（3SN） | `configs/384card_3sn/pf3_3sn_dp128_edp3.yaml` | **18.1459 s** | 166,342 | 39.19 GB |
| 128 卡（1SN） | `configs/128card_1sn/ep64_oswap_st128.yaml` | **21.38 s** | ~49,000 | 45.88 GB |

2SN 冠军相对旧基线（GBS 256 = `pf3_oswap_st128.yaml`）**每样本 −10.4%**，MFU 25.13% → **28.05%**。

### 1.2 2SN GBS 阶梯（同一窗口，其余配置完全相同）

| GBS | micro | 配置文件 | min 步时 | ms/样本 | 等效 256 | padded tok/s | MFU | 峰值 |
|---|---|---|---|---|---|---|---|---|
| 256 | 1 | `256card_2sn/pf3_oswap_st128.yaml` | 20.4957 s | 80.06 | 20.50 s | 102,322 | 25.13% | 41.49 GB |
| 512 | 2 | `256card_2sn/gbs512_rbwd_true.yaml` | 38.5778 s | 75.35 | 19.29 s | 108,723 | 26.71% | 48.87 GB |
| 1024 | 4 | `256card_2sn/gbs1024.yaml` | 75.0517 s | 73.29 | 18.76 s | 112,088 | 27.53% | 48.91 GB |
| 2048 | 8 | `256card_2sn/gbs2048_rbwd_true.yaml` | 148.5950 s | 72.56 | 18.57 s | 112,900 | 27.73% | 48.97 GB |
| **4096** | **16** | `256card_2sn/gbs4096_rbwd_true.yaml` | **293.8240 s** | **71.73** | **18.364 s** | **113,400** | **28.05%** | **49.10 GB** |

4 点最小二乘拟合 **`step = 71.524 ms × GBS + 2.017 s`**（偏差 ≤0.83%），渐近线 **18.310 s**：

- **收益全部来自摊薄每步固定开销 `b`**；每样本成本 `a` 不随 GBS 变 ⇒ **加大 GBS 没有藏住任何通信**。
- 4096 距渐近线只剩 0.3% ⇒ **GBS 杠杆已吃干**，后续优化必须落在 `a ≈ 71.7 ms/样本` 上。
- 峰值从 2 micro 起就持平 48.9–49.1 GB（距 ~61.3 GB 上限仍有 ~12 GB）。

> 这三个 `gbs*_rbwd_true.yaml` 名字里的 `rbwd_true` 是**必须**的：`reshard_after_backward: false`
> 在累积 >1 时会让全量参数常驻，step 0 直接 OOM。包里所有配置都已是 `true`。

### 1.3 复现判据（2026-09-21 复核，`r36_champ100`，100 步请求 / 实际 57 步数据耗尽即停）

| 指标 | 值 |
|---|---|
| min / median / max / mean 步时 | 293.971 / 316.282 / 331.363 / 313.731 s |
| 每样本 | 71.770 ms ⇒ **等效 GBS-256 = 18.3732 s**（与记录的 18.364 s 差 **0.05%**） |
| 峰值 allocated | 49.099 GB |
| MFU / HFU | 0.2804 / 0.3739（HFU = MFU × 4/3，因为 `activation_checkpoint.mode: full`） |

⚠️ 注意 **均衡 router 下 max/min 仍达 1.127×** —— 抖动来自集群/调度，不是负载不均。
真实路由（不强制均衡）的对照配置是 `256card_2sn/gbs4096_realrouter.yaml`。

### 1.4 真实数据线（COCO2017 8K packing，另一条口径）

`configs/realdata_2sn/coco2017_2sn_8k_packed_cap45.yaml`：`OmniPackingLoader` 按 8K
token 预算选样本，`SamplePacker` 顺序拼接整个样本（不再把窗口补到 8192，而是拼到预算内，
并下发 int32 `cu_seq_lens` 做块对角注意力）。该配方实测：稳态 **50–58 s/step**、
峰值 **50.37 GB**、**real tok/s 8,210**、supervised/padded 22.6%；
等步数 loss 下降是 padded 臂的 **1.89×**。

> 这条线的步时**不能**和 §1.1/§1.2 的合成数据吞吐数字直接比较（序列长度分布完全不同）。

---

## 2. 包内文件

| 包内路径 | 原始路径（作者工作区） | 作用 |
|---|---|---|
| `configs/256card_2sn/gbs4096_rbwd_true.yaml` | `fam256loc/gbs4096_rbwd_true.yaml` | **2SN 冠军**（GBS 4096） |
| `configs/256card_2sn/gbs2048_rbwd_true.yaml` | `fam256loc/gbs2048_rbwd_true.yaml` | 阶梯 2048 |
| `configs/256card_2sn/gbs1024.yaml` | `fam256loc/gbs1024.yaml` | 阶梯 1024 |
| `configs/256card_2sn/gbs512_rbwd_true.yaml` | `fam256loc/gbs512_rbwd_true.yaml` | 阶梯 512 |
| `configs/256card_2sn/pf3_oswap_st128.yaml` | `fam256loc/pf3_oswap_st128.yaml` | 2SN 旧基线（GBS 256，A/B 对照用） |
| `configs/256card_2sn/gbs4096_realrouter.yaml` | `fam256loc/gbs4096_realrouter.yaml` | 真实路由对照（不强制均衡） |
| `configs/512card_4sn/pf3_dp128.yaml` | `4snr6/pf3_dp128.yaml` | 512 卡冠军（`dp_shard_size: 128`） |
| `configs/384card_3sn/pf3_3sn_dp128_edp3.yaml` | `3sn/pf3_3sn_dp128_edp3.yaml` | 384 卡冠军（`edp_shard_size: 3`） |
| `configs/128card_1sn/ep64_oswap_st128.yaml` | `fam128/ep64_oswap_st128.yaml` | 128 卡冠军（ep64 / edp2） |
| `configs/realdata_2sn/coco2017_2sn_8k_packed_cap45.yaml` | `coco2017_2sn_8k_packed_cap45.yaml` | 真实数据 8K packing 线 |
| `reports/CURRENT_BEST.md` | `CURRENT_BEST.md` | **成绩单**：全部数字 + 已封档方向表 |

所有 YAML 都是**自包含**的（没有 `defaults:` / include），可以直接单独使用。

### 各配置关键参数

| 配置 | GBS | dp_shard | edp_shard | AC | opt swap | reshard_after_backward |
|---|---|---|---|---|---|---|
| `gbs4096_rbwd_true`（2SN 冠军） | 4096 | 128 | 2 | full | on / st128 | true |
| `gbs2048_rbwd_true` | 2048 | 128 | 2 | full | on / st128 | true |
| `gbs1024` | 1024 | 128 | 2 | full | on / st128 | true |
| `gbs512_rbwd_true` | 512 | 128 | 2 | full | on / st128 | true |
| `pf3_oswap_st128`（2SN 基线） | 256 | 128 | 2 | full | on / st128 | true |
| `gbs4096_realrouter` | 4096 | 128 | 2 | full | on / st128 | true |
| `pf3_dp128`（512 卡） | 512 | 128 | 4 | full | off | true |
| `pf3_3sn_dp128_edp3`（384 卡） | 384 | 128 | 3 | full | off | true |
| `ep64_oswap_st128`（128 卡） | 128 | 128 | 2 | full | on / st128 | true |
| `coco2017_..._packed_cap45`（真实数据） | 256 | 256 | 2 | full | off | true |

---

## 3. 前置条件

1. **代码**：本 PR 所在的 commit（配置的 `_target_` 指向仓库内的模块 ——
   `examples.training_demo.cropped_kimi_vlm.build_cropped_kimi_vlm`、
   `hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn`、
   `hyper_parallel.data.omni.*`、`hyper_parallel.components.optim.builders.AdamW`）。
2. **模型目录**：见 §4。**只读 `config.json` 与 processor/tokenizer 文件**；
   这些吞吐配置**都不加载 1.9T 权重**（没有任何一个设 `load_pretrained: true`）。
3. **数据**：见 §4。合成吞吐数据的 JSON 需**样本数 ≥ `global_batch_size`**，
   否则训练中途数据耗尽（`train_iters` 语义见 `training.train_iters`）。
4. **环境**：CANN 环境脚本 + conda env `hyper-parallel`；建议 `HF_HUB_OFFLINE=1`、
   `HCCL_CONNECT_TIMEOUT=1800`、`HCCL_EXEC_TIMEOUT=1800`。
5. **节点**：`节点数 = 卡数 / 8` —— 128 卡 16 节点、256 卡 32 节点、384 卡 48 节点、512 卡 64 节点。
   **拉起前务必确认节点真空闲**：本集群别人的任务常常不是 `torchrun` 形式，
   仅看进程名会误判，随后我们的作业会以"诡异的低分配 OOM"失败。
   仓库自带工具：`bash /home/fdd/workspace/bin/find_empty_nodes.sh <ip_list_file>`。
6. **HF `datasets` 缓存必须先预热**（2026-09-23 实测，冷缓存下 256 rank 必挂）。
   omni 数据层经 `datasets.load_dataset` 读源，缓存默认落在 `~/.cache/huggingface/datasets`。
   冷缓存时 256 个 rank 会同时构建同一个缓存条目，而 HF 的 builder 锁在本集群共享文件系统上
   **无法串行化加载后的 `.filter(sample_filter)` 这一步**，于是索引缓存被并发写坏，全部 rank 报：

   ```
   FileNotFoundError: .../datasets/json/default-<hash>/.../cache-<hash>.arrow
   OSError: error stat()ing file
   ```

   必须先**单进程**把两级缓存建好，再用 `HF_DATASETS_CACHE` 指向它启动：`json-train.arrow`
   （源渲染）与 `.filter()` 的 `cache-<hash>.arrow`（行索引）。注意第二级必须用**同一个函数对象**
   （`OmniDataTransform.is_valid_sample`，指纹会哈希它），所以要在 `hyper_parallel` 可导入的环境里建
   （本机裸 Python 导入会因 torch_npu 初始化卡住，用 pytest 跑一个一次性模块最稳）。
   这是 omni 加载器的性质，不是本包特有：trainer_dev 自己的 omni 配置冷缓存同样会撞。
7. **真实路由那一档必须配 expert capacity**。`configs/256card_2sn/gbs4096_realrouter.yaml`
   （`fix_router: false`）交付的形态是 `swap_inputs: false` + 无 `capacity_factor`，
   正是文档里 `r37` 记录的 OOM 形态：真实路由下最忙的 rank 承担峰值，GBS4096 会
   `NPU out of memory`。实测即使补上 `--activation_checkpoint.swap_inputs=true`，GBS256 仍
   OOM（已分配 54.10 GB / 仅剩 4 GB）；**加上 `HP_EP_CAPACITY_FACTOR=4.5` 后正常跑完**
   （8 步 0 报错，峰值 52.90 GB，min 步时 40.45 s）。均衡档（`fix_router: true`）不需要 capacity，
   它靠"每 rank token 数完全相同"压内存。

---

## 4. 需要改的路径

配置里的绝对路径全部指向作者的集群存储，**换机器必须改**：

| YAML key | 当前值 | 说明 |
|---|---|---|
| `model.config_path` | `/home/fdd/workspace/models/moonshotai/Kimi-K2.6` | Kimi-K2.6 模型目录。`AutoConfig.from_pretrained` 从这里读 `config.json`（`model_type` 必须是 `kimi_k25`） |
| `model.pretrained_model_name_or_path` | 同上 | 模型加载路径 |
| `dataset.model_assets.pretrained_model_name_or_path` | 同上 | VLM trainer 用 `dataset.model_assets`（`hyper_parallel.data.omni.omni_transform.build_auto_processor`）并透传 `model.trust_remote_code=false`，从这里建 native processor |
| `dataset.data_path`（9 个配置） | `/home/fdd/workspace/datasets/mm_datasets/mock_vlm_dataset/mock_data_pic_num_10_textlen_10240.json` | 合成吞吐数据（457 MB 目录，10240 条 × 10 图） |
| `dataset.data_path`（COCO 线） | `/home/fdd/workspace/datasets/COCO2017/mllm_format_llava_instruct_data.json` | 真实数据（196 MB） |

替换示例（两个 key 一起改）：

```bash
sed -i 's#/home/fdd/workspace/models/moonshotai/Kimi-K2.6#/your/path/to/Kimi-K2.6#g' \
       examples/training_demo/perf_kimi_k26_2sn/configs/256card_2sn/gbs4096_rbwd_true.yaml
```

数据格式是 content-list（媒体位置显式），图片路径相对 JSON 所在目录解析：

```json
[{"messages": [
    {"role": "user", "content": [
      {"type": "image", "url": "images/a.png"},
      {"type": "text",  "text": "Describe the following image."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}],
  "images": ["images/a.png"]}]
```

---

## 5. 启动

入口必须是 **`scripts/train_vl.py`**（VLM 专用；`train_lm.py` 是纯文本入口，跑不了 VLM）。
每个节点各起一个 `torchrun`，`--node_rank` 各不相同：

```bash
cd <repo>

# 0) 先确认节点真空闲（见 §3.5）
bash /home/fdd/workspace/bin/find_empty_nodes.sh my_workspace/ip_train.txt

# 1) 每个节点上执行（RANK / MASTER_ADDR 按你的调度填）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800

torchrun \
  --nproc_per_node=8 \
  --nnodes=32 \
  --node_rank=<RANK> \
  --master_addr=<MASTER_ADDR> \
  --master_port=<PORT> \
  scripts/train_vl.py \
  examples/training_demo/perf_kimi_k26_2sn/configs/256card_2sn/gbs4096_rbwd_true.yaml
```

`--nnodes` 按 §3.5 的规模填（2SN = 32）。

### ⚠️ 只跑 6 步：测 min 步时前先覆盖 `train_iters`

包里的**合成吞吐配置都写死 `train_iters: 6`** —— 拉起后跑 6 步就停。这够验证"能不能拉起来"，
但**不足以量出稳定的 min 步时**。要复现 §1.3 的数字，把步数覆盖掉
（09-21 的复核 `r36_champ100` 就是请求 100 步、数据耗尽后实际 57 步）：

```bash
torchrun ... scripts/train_vl.py \
  examples/training_demo/perf_kimi_k26_2sn/configs/256card_2sn/gbs4096_rbwd_true.yaml \
  --training.train_iters=100
```

覆盖语法是 `--<点分路径>=<值>`（如 `--accelerator.tp_size=4`），可追加任意多个。
真实数据那条线（`coco2017_2sn_8k_packed_cap45.yaml`）本身写的就是 `train_iters: 1800`。

**跑之前建议先跑 A/B 对照**：同一批节点、同窗口交错跑冠军与
`pf3_oswap_st128.yaml`（GBS 256 基线），否则单臂数字会被集群抖动吃掉
（均衡档 max/min 仍有 1.127×）。

---

## 6. 口径与注意

1. **合成数据 vs 真实数据**：§1.1/§1.2 是固定长度合成数据的吞吐口径（`mock_vlm_dataset`）；
   §1.4 是真实 COCO packing 口径，两者**不可直接比较**。
2. **MFU / HFU**：MFU 用有用 FLOPs（6N）除以峰值，**不含**重算；HFU 把重算也算进去，
   所以 `activation_checkpoint.mode: full` 时 `HFU = MFU × 4/3`。
3. **padded token 口径**：tok/s 按 **padded** 序列槽位算（硬件实际算的），不是有效 token 数。
4. **`fix_router: true`** 是强制均衡口径；真实路由见 `gbs4096_realrouter.yaml`。
5. **2SN 上 `optimizer.swap` 是刚需**：不开swap 需要 56.59 GB 且步时在 24.9–41.0 s 之间剧烈抖动；
   开 swap 后 41.49 GB / 21.06–22.84 s 稳定。注意这与 **512 卡档的结论相反**
   （那里 swap 是 −9.95 GB 换 +7.3% 步时），**不要跨规模外推**。
6. 所有配置都要求 `reshard_after_backward: true`（见 §1.2 的注）。
7. **步数**：合成吞吐配置只跑 6 步（`training.train_iters: 6`）；要测 min 步时请按 §5 加
   `--training.train_iters=100`，否则 6 步里任何一次抖动都会直接进入结果。

---

## 7. 已封档的方向（不要再投入）

完整表见 `reports/CURRENT_BEST.md` §"已封档的方向"。几条最容易重复踩的：

| 方向 | 结论 |
|---|---|
| **a2a 层次化重构（DeepEP 式）** | 位精确但**慢 29.9%**（42.767 → 61.017 ms）；问题是没被 overlap 藏住，不是算法或带宽 |
| **`torch.compile`（NPU 后端）** | 融合确实发生（7088 个 kernel），但 min 步时 **+0.78%**，且首步多 26.5 min 编译 |
| **selective AC 省重算** | 不存在既装得下又有收益的 `N`（`N=120` 与全 AC 逐位相同，`N=200` OOM） |
| **MoE dispatch 静态计划（`HP_EP_STATIC_SPLITS`）** | **默认关且已标 BLOCKED**：全量 4/4 OOM vs 对照 0/2，定因是两次 `.tolist()` drain 兼作主机侧节流阀 |
| **`HP_EP_FUSED_DISPATCH`** | +37.7%，且有静默断梯度的陷阱 |
| **`HP_EP_DISPATCH_CHUNKS` 2/4** | 更慢；`HP_EP_EQUAL_A2A` 中性；`HP_EP_A2A_SINGLE` 对本模型是死代码 |
| **GBS 杠杆** | 距渐近线 0.3%，吃干 |
| **TP=2 / CP=2 / seq 形状 / FSDP `enable_offload`** | 更差或 OOM（详见成绩单表） |

---

## 8. 未随包提供的材料

`reports/CURRENT_BEST.md` 正文里引用的这些文档**留在作者工作区**，未包含在本 PR 中：

- `overnight_optimization_20260916.md`（512 卡夜间过程）
- `opt_loop_2sn_20260920.md`（2SN 单日循环全过程，247 KB）
- `NIGHTLY_SUMMARY_20260920.md`（Round-23→35 夜间测试）
- `reshard_after_backward_landmine_20260920.md`（`reshard_after_backward` 地雷）
- `PERF_INVALID_ATTEMPTS_20260920.md` / `PERF_GAINS_WITH_EVIDENCE_20260920.md`
- `COCO8K_PACKED_RUN.md`（真实数据线细节）
- `2SN_OPTIMIZATION_SUMMARY.md`（2SN 汇总）

以及工具脚本（`ab_verdict.py`、`prof_ledger.py`、`run_arm_guarded.sh`、`arm_watchdog.sh` 等）
和原始 profile / 日志（体积过大）。需要的话在 PR 里说一声，我可以单独打包。
