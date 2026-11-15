# 环境审计

## 当前电脑（2026-11-15）

- Windows GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，驱动 617.14。
- WSL：Ubuntu 24.04.3 LTS，kernel `6.6.87.2-microsoft-standard-WSL2`。
- WSL Python：3.12.3。
- WSL 可访问同一块 NVIDIA GPU。
- WSL 系统 Python 仍保持不变；任务专用前缀
  `/home/s/.local/share/flowpolicy/miniforge-py310` 使用 Python 3.10.14。
- 该前缀已安装 Jittor 1.3.11.0、NumPy 1.26.4、Zarr 2.12.0、
  PyYAML 6.0.2 和 pytest 7.4.4，当前只验证 CPU backend。
- WSL 没有 CUDA Toolkit；训练环境不安装 MuJoCo，仿真依赖位于单独的 `sim-env`。
- WSL 根分区可用空间约 870 GB。

## 与上游环境的差异

上游安装文档的测试环境是 Linux、Python 3.8、CUDA 11.7、MuJoCo 2.1、mujoco-py 2.1.2.14 和 Gym 0.21。当前项目没有静默声称使用了这套旧环境：实际采用现代替代栈，并在 D016 中记录深度、裁剪框和 API 差异。

因此采用两个隔离环境：

| 环境 | 路径与用途 | 关键依赖 |
|---|---|---|
| `flowpolicy-jittor` | `/home/s/.local/share/flowpolicy/miniforge-py310`；数据、训练、推理 | Python 3.10.14、Jittor 1.3.11.0、NumPy 1.26.4、Zarr 2.12.0、PyYAML 6.0.2 |
| `sim-env` | `/home/s/.local/share/flowpolicy/sim-env`；示范、EGL 渲染、rollout | Python 3.10.14、MuJoCo 3.3.7、MetaWorld 3.0.0、Gymnasium 1.4.0、Zarr 2.12.0 |

两个环境通过 NumPy/Zarr 和 JSON 行协议交互；训练计算图中没有 PyTorch。点云 FPS 由
`scripts/record_demos.py` 的确定性 NumPy 实现完成；PyTorch 只存在于独立 parity-env，用于数值对照。

正式配置前可运行：

```bash
source /home/s/.local/share/flowpolicy/miniforge-py310/bin/activate
python scripts/check_environment.py
python scripts/check_environment.py --strict-train --allow-cpu
python scripts/check_environment.py --strict-train
```

不激活 shell 时，也可直接调用本机任务专用解释器：

```bash
/home/s/.local/share/flowpolicy/miniforge-py310/bin/python \
  scripts/check_environment.py --strict-train --allow-cpu
```

第一条只报告当前解释器的包元数据。训练严格模式会在隔离子进程中真正导入依赖：
`--strict-train --allow-cpu` 验证本机 CPU 开发环境；`--strict-train`
还检查 Python 版本，并实际同步执行最小 CUDA 运算，而不只是读取
`jt.has_cuda`。当前 `--strict-sim` 保留为**上游旧版 Python 3.8/mujoco-py 栈的兼容性审计**，
不用于证明现代 `sim-env` 可用；现代栈的 EGL、RGB、depth、点云和专家动作已经通过真实示范采集
与闭环 rollout 验证，原始记录见 `env/sim_setup_log.md` 和 `env/metaworld_contract.md`。

严格模式首次导入 Jittor 会触发 JIT 编译，也可能由 Jittor 自身下载
oneDNN/MKL 运行库，因此并非无副作用。当前机器的 Jittor 1.3.11.0 已缓存：

```text
~/.cache/jittor/mkl/dnnl_lnx_2.2.0_cpu_gomp.tgz
MD5:    35bbbdf550a9d8ad54db798e372000f6
SHA256: 06d45ebf9cde5d3dd815c9ec3ad74cd10f0f49294c04cda87fe80bd99ad67dee
```

MD5 与 Jittor 1.3.11.0 源码内置校验值一致。完全离线或新云主机运行时，
需要预置这一缓存或事先允许 Jittor 完成首次下载；不能把“Python 包离线
安装完成”误写成“首次运行也完全不联网”。如首次编译较慢，可显式增加
探测超时，例如 `--probe-timeout 1200`。

## 为什么不使用系统 Python

WSL 系统 Python 是 3.12.3，而 Jittor 与仿真依赖需要稳定、可冻结的版本组合。
训练和仿真分别使用隔离的 Python 3.10.14，避免包冲突，也便于判断故障来自模型还是仿真。
旧版 Python 3.8/mujoco-py 路线仅作为上游对照，不是当前已运行环境。

## 阿里云 A10 环境（2026-11-15）

PAI-DSW 公共资源实例已经补上本机缺失的 CUDA 证据：NVIDIA A10 23028 MiB、驱动
570.211.01、CUDA 12.1.105、Python 3.11.11、Jittor 1.3.11.0。`--strict-train`
实际执行 CUDA kernel 并通过；完整 `512/1024/2048` 通道在 1024 点、batch 1 下完成一次
forward/backward/update，峰值显存约 8.3 GiB。

在资源探测之后，完整通道模型又以 batch 64、30 epochs、1050 updates 完成 3 个独立训练 seed。
每个 seed 约 688 秒，外部监控峰值 10726 MiB；三个最后 epoch mean loss 的均值为
`6.819e-6`。这是论文网络结构的有限预算试验，仍不同于论文 3000 epochs、旧仿真栈和
detached EMA target 配方。

云端仍保持训练与仿真隔离：`/mnt/workspace/venvs/flowpolicy` 只放 Jittor 训练依赖，
`/mnt/workspace/venvs/flowpolicy-sim` 放 MuJoCo 3.3.7、Meta-World 3.0.0 和 Gymnasium 1.4.0。
详细路径、命令与结果见 `docs/CLOUD_ALIYUN.md`。

