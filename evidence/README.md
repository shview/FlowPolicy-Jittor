# 公开证据索引

本目录只保存可以公开复核的 JSON 和 Markdown。原始 Zarr 数据、模型权重、trainer checkpoint、
完整云端日志和账号信息不进入 Git。运行以下命令可以重新检查 manifest、逐回合结果、均值、样本
标准差、UTF-8 编码和隐私边界：

```bash
python scripts/verify_public_evidence.py
```

## 结果与证据的对应关系

| 阶段 | 要回答的问题 | 已验证结果 | 入口文件 | 不能据此声称 |
|---|---|---|---|---|
| FPJ-032 | Jittor CUDA、完整网络和 A10 闭环能否运行 | CUDA 与完整通道有限预算运行通过；Button Press Wall 三 seed 为 100% | [`README.md`](../README.md)、[`docs/CLOUD_ALIYUN.md`](../docs/CLOUD_ALIYUN.md) | 37 任务或论文 3000 epochs 已复现 |
| FPJ-033 preliminary | 更难任务在上云前是否通过本地门槛 | 专家、数据指纹、训练与 5 回合 smoke 通过 | [`fpj033/preliminary/manifest.json`](fpj033/preliminary/manifest.json) | 完整通道正式成功率 |
| FPJ-033 formal | 完整网络在 Reach Wall 的三 seed 有限预算表现 | 4/20、5/20、10/20，即 31.67% ± 16.07% | [`fpj033/formal/manifest.json`](fpj033/formal/manifest.json) | 与论文旧仿真栈或论文统计口径直接等价 |
| FPJ-034 | detached EMA target 是否真正冻结，以及小配置资源开销 | target 可训练参数为 0；EMA 更新步为 8；构造 RSS 比无 EMA 多 3.75 MiB | [`fpj034/resource_probe.json`](fpj034/resource_probe.json) | 完整网络成功率或 GPU 资源结论 |
| FPJ-035 | 相同小通道短预算下两种 target 语义有何差异 | code 51.67% ± 5.77%；detached 16.67% ± 15.28% | [`fpj035/manifest.json`](fpj035/manifest.json) | detached target 普遍无效 |
| FPJ-036 | 两种 target 能否在完整网络 A10 上端到端运行 | seed 0：code 10/20，detached 6/20；匹配检查通过 | [`fpj036/manifest.json`](fpj036/manifest.json) | 单 seed 可以给出均值、标准差或稳定排名 |
| FPJ-037 | 完整通道三 seed 下两种 target 是否有稳定差异 | code 28.33% ± 20.21%；detached 26.67% ± 20.21%；仅差 1 个成功回合 | [`fpj037/manifest.json`](fpj037/manifest.json) | 两种 target 已有统计显著差异，或诊断 rollout 是新训练 seed |

FPJ-032 发生在统一公开 evidence bundle 机制建立之前，因此没有单独的 `fpj032/` 目录；它的配置、
结果边界和云端环境记录保留在 README、复现计划、云端记录和串行 gate 记录中。后续正式 bundle 从
FPJ-033 开始。

## 完整通道三 seed target 结果

FPJ-037 复用 FPJ-036 已发布的 seed 0，并补跑完整通道 seeds 1/2。四个新增训练单元、80 个新增
闭环 episode、checkpoint 哈希和配对配置均已独立校验：

| target | seed 0 | seed 1 | seed 2 | 三 seed 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| code | 10/20 | 5/20 | 2/20 | **28.33% ± 20.21%** |
| detached | 6/20 | 9/20 | 1/20 | **26.67% ± 20.21%** |

两种模式在 60 个 episode 中分别成功 17 次和 16 次，样本标准差相同且很大，因此这里支持的结论是
“短预算下没有稳定排名”，而不是 code 或 detached 普遍更好。新结果单独发布到 `evidence/fpj037/`，
没有改写已经发布的 `evidence/fpj036/`。

同一 seed-1 trainer 的评测诊断另存于 `diagnostics.json`：code 在线/code 为 5/20，code EMA/code 为
2/20，code 在线/exact 为 3/20；detached 在线/code 为 9/20，detached EMA/code 为 10/20。这些结果
只用于解释权重选择和动作切片，不计入上表的独立训练 seed，也不能从一个 seed 推广为普遍结论。

旧证据的 manifest 在 Windows CRLF 工作区生成，而 Git 保存的是规范化 LF；这些已发布的 manifest
不能改写。校验器只对历史 JSON/Markdown 接受 LF 与 CRLF 的等价转换，并仍要求转换后哈希和字节数
完全匹配。新建证据文件由 `.gitattributes` 统一为 LF，哈希直接按仓库字节计算。

## 统一解释边界

- 成功率来自闭环 episode，不从训练 loss 推导。
- `±` 使用三个训练 seed 的样本标准差；一个 seed 时保持 `null` 或明确写成单次观察。
- 30 epochs 只是有限预算。论文使用 3000 epochs、每 200 epochs 评测并取最高五次成功率均值。
- 现代 MetaWorld/MuJoCo 是替代仿真栈，任务名相同不代表观测和动力学逐帧一致。
- NFE=1、CUDA 可运行和低延迟可以分别陈述，但不能合并成“论文成功率已复现”。
