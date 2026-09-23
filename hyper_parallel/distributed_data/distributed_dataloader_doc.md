# Distributed dataloader

## 1. 使用说明

使能 dataloader，需要先定义每个 sample 的 metadata。例如，多模态样本需要提供
文本和图像部分的 seq length，供 cost model 计算。默认模型中，`P` 表示 AR 前缀长度
（含文本、控制符和条件图像 token），`D` 表示生成图像的 latent token 数。

然后构建 `DistributedDatasetConfig`，主要配置：

- `seq_len`：一个 packed sample（pack）的最大序列长度。
- `local_batch_size`：每个 rank 每次取出的 microbatch 数。
- `communication_backend`：节点内 DP 数据交换使用 `hccl` 或 `gloo`，默认 `hccl`。
- `dp_balance_log`：`0` 关闭、`1` 开启均衡日志，默认 `1`；不影响均衡本身。

其次，通过 `build_distributed_dataset` 绑定：

- `source`：用户现有的数据来源，负责读取、预处理、采样和初步 packing。
  它是每个 rank 上的可迭代对象，每次提供一步尚未 collate 的原始样本，
  结构为 `[[sample, ...], [sample, ...], ...]`，外层长度为 `local_batch_size`。
- `metadata`：刚才定义的 `sample -> SampleMetadata` 函数。
- `collate_fn`：用户模型自己的 collator，在均衡后把一个 pack 内的 samples 组装成模型输入。
- `log_fields`：希望在日志中暴露的 metadata 数值字段，例如 P/D、图片数量等，按 pack 求和。
- `cpu_fields`：collate 输出中需要保留在 CPU 的字段，例如 `cu_seqlens`。
  均衡计算读取的是 metadata；`cpu_fields` 控制字段是否上卡。

最后，用 dataset、config 和并行 mesh 构建 dataloader：

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataset,
    build_distributed_dataloader,
)


def sample_metadata(sample):
    prefix = sample["prefix_tokens"]
    image = sample["diffusion_tokens"]
    return SampleMetadata(
        pack_tokens=prefix + image,
        features={
            "P": prefix,
            "D": image,
            "cond_image_token_lengths": sample.get("cond_image_token_lengths", []),
        },
    )


config = DistributedDatasetConfig(
    seq_len=16384,
    local_batch_size=2,
    dataset_already_sharded=True,
    communication_backend="gloo",
    dp_balance_log=1,
)
dataset = build_distributed_dataset(
    source,
    metadata=sample_metadata,
    collate_fn=model_collator,
    cpu_fields=("cu_seqlens",),
    log_fields=("P", "D"),
)
loader = build_distributed_dataloader(
    dataset, mesh, config,
    model_config=model_config,  # 默认 cost 所需的实际模型配置
    device=device,             # 当前 rank 的 NPU/CUDA device
    max_steps=train_steps,
)
```

返回的 dataloader 是可迭代对象，连续提供均衡、collate 和 H2D 后的 batch。
每次迭代返回 `local_batch_size` 个 packed microbatch。

## 2. 如何自定义 DP balance 的 cost

用户需要定义一个根据 `SampleMetadata` 估计 cost 的函数，在 `__call__` 中计算 score，
并返回 `WorkloadCost(llm=score)`。默认均衡算法使用其中的 `llm` 分量。

Dataloader 自动将一个 pack 内所有 sample 的 cost 相加，再将每个 rank 这一步的
所有 pack cost 相加，用于 rank 间均衡。

### 2.1 默认 cost

不传 `cost_model` 时，使用 `DefaultCostModel(model_config)`。
默认 score 以 backbone FLOPs 为基础，计算线性层（Q/KV/O 投影、dense/MoE MLP）
和 attention 的成本：

```text
sample_cost = training_multiplier * (linear_flops + attention_flops)
attention_area ≈ 0.5 * P² + D * (P + D)
```

文字采用常规 causal 下三角；生成图像采用图像块内 full attention，并能看到前缀。
若前缀中的 condition image 也使用 full attention（`blockmask`），再增加
`0.5 * SUM(condition image block length²)`。独立 sample 之间没有 attention，因此先算各自 cost 再求和。
线性层和 attention 系数从模型配置计算，默认 `training_multiplier=3`；
该 cost 仅估计 backbone，不包含 ViT、VAE。

### 2.2 自定义 WorkloadCost：以 seq length 为 cost

例如在 `my_cost_model.py` 中定义：

```python
from hyper_parallel.distributed_data import SampleMetadata, WorkloadCost


class SeqLengthCost:
    model_id = "seq-length-v1"

    def __call__(self, metadata: SampleMetadata) -> WorkloadCost:
        score = float(metadata.pack_tokens)
        return WorkloadCost(llm=score)
```

这个例子只按 sample 的总序列长度估价，不区分 attention mask。
从自定义 Python 文件导入实例，通过 `cost_model` 传给 builder：

```python
from my_cost_model import SeqLengthCost

loader = build_distributed_dataloader(
    dataset, mesh, config, device=device,
    cost_model=SeqLengthCost(),
)
```

## 3. 如何自定义均衡算法和优化目标

均衡算法需要定义两个函数：

- `objective(rank_costs)`：设置优化目标，返回越小越好的 score。
  默认是 `makespan`，即最慢 rank 的 cost：`max(rank_cost.llm)`。
- `assign(...)`：设置样本分配策略。默认是 LPT 贪心分桶：按 sample cost 从大到小，
  分配到容量允许、当前 rank 总 cost 最低的 bin。

默认配置等价于：

```python
from hyper_parallel.distributed_data import LPTBalancingAlgorithm

balancing_algorithm = LPTBalancingAlgorithm(objective="makespan")
```

Hyper 用同一个 objective 比较原排布和候选排布；相对改善严格大于
`config.min_balance_gain`（默认 0）才采用，否则保留原排布。

### 3.1 自定义示例：蛇形分配 + balance 目标

在 `my_balancing_algorithm.py` 中定义下面两个方法。
`assign` 使用已经计算好的 sample cost，按降序在 bins 之间交替正向、反向分配；
`objective` 使用归一化方差衡量各 rank 的负载差异，越小越均衡。

```python
class SnakeBalanceAlgorithm:
    algorithm_id = "snake-balance-v1"
    objective_name = "balance"

    def objective(self, rank_costs):
        maximum = max((cost.llm for cost in rank_costs), default=0.0)
        if maximum == 0:
            return 0.0
        loads = [cost.llm / maximum for cost in rank_costs]
        mean = sum(loads) / len(loads)
        return sum((load / mean - 1.0) ** 2 for load in loads) / len(loads)

    def assign(self, samples, *, reference_bins, constraints,
               data_parallel_size, local_batch_size):
        count = data_parallel_size * local_batch_size
        bins = [[] for _ in range(count)]
        tokens = [0] * count
        budgets = [{} for _ in range(count)]
        ordered = sorted(samples, key=lambda item: (-item.metadata.cost.llm, item.key))

        for index, item in enumerate(ordered):
            order = list(range(count))
            if (index // count) % 2:
                order.reverse()
            start = index % count
            for target in order[start:] + order[:start]:
                if index < count and bins[target]:
                    continue  # 第一轮保证每个 bin 非空
                if constraints.fits(tokens[target], budgets[target], item):
                    bins[target].append(item.key)
                    tokens[target] += item.metadata.pack_tokens
                    budgets[target] = constraints.add_costs(budgets[target], item)
                    break
            else:
                return reference_bins  # 装不下时保留原排布
        return bins
```

返回的 bins 按 rank、再按 microbatch 顺序排列，包含每个输入 sample 的 key，
且每个 key 恰好出现一次。每个 bin 必须非空，并满足 packing 容量限制。

将 `objective` 和 `assign` 放在同一个算法类中，从自定义 Python 文件导入实例，
通过 `balancing_algorithm` 传给 dataloader；它与 `cost_model` 可以独立替换：

```python
from my_cost_model import SeqLengthCost
from my_balancing_algorithm import SnakeBalanceAlgorithm

loader = build_distributed_dataloader(
    dataset, mesh, config, device=device,
    cost_model=SeqLengthCost(),
    balancing_algorithm=SnakeBalanceAlgorithm(),
)
```
