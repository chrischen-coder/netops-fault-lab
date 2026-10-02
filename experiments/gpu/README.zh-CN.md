# GPU 环境与训练自检

[English](README.md)

这里验证模型能否下载、GPU 能否计算、梯度能否同步，以及 LoRA 能否更新并重新加载。
业务准确率仍以[故障定位实验](../../docs/experiments.md)为准；两个图文自检题不构成模型评测。

## 安装

Linux x86_64、Python 3.12、支持 CUDA 12.8 的 NVIDIA 驱动。RTX 5090 需要包含
Blackwell 支持的构建。本目录固定 PyTorch 2.8.0 的 CUDA 12.8 wheel；不依赖系统
`nvcc` 的版本，也不需要更改全局 Python 或重新安装驱动。
[PyTorch 版本表](https://pytorch.org/get-started/previous-versions/) ·
[Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。

从仓库根目录执行。以下命令使用已有的 `uv`；虚拟环境请选择新目录。

```bash
uv venv --python 3.12 runs/gpu-venv
uv pip install --python runs/gpu-venv/bin/python --link-mode copy \
  -r experiments/gpu/requirements-cuda.txt \
  -c experiments/gpu/requirements-linux-py312.lock \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python runs/gpu-venv/bin/python --link-mode copy \
  -r experiments/gpu/requirements.txt \
  -c experiments/gpu/requirements-linux-py312.lock \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

也可以用 `python3.12 -m venv runs/gpu-venv`，再用该环境的 `python -m pip install`
执行相同的两个 requirements 安装命令，去掉 `--python` 和 `--link-mode`。
PyTorch wheel 从官方 CUDA 源下载，其他 Python 包使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。

## 模型下载

采用 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)，
固定 Hugging Face revision：`89644892e4d85e24eaac8bacfd4f463576704203`。
模型本身使用 Apache-2.0 许可证，权重不放进本仓库。

Hugging Face 无法直连时，从 Qwen 在[魔搭的官方仓库](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct)下载：

```bash
python3 experiments/gpu/download_model.py \
  --source modelscope --output runs/models/Qwen3-VL-2B-Instruct
```

需要 `curl`。下载支持断点续传，每个文件完成后检查大小和 SHA-256。
[清单](model-manifest.json)记录两个平台的 revision，以及同一套权重、tokenizer 和配置的哈希。
已有但不匹配的文件会报错，不会被覆盖。可以改用 `--source huggingface` 下载同一版本。

权重约 4.26 GB；另需为 CUDA/PyTorch 安装包、缓存和环境预留磁盘空间。
下载完成后，下面两个检查可以离线运行，不需要 API key。

## 八卡检查

先确认这些 GPU 空闲，再运行；较少显卡时调整可见设备和进程数。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  runs/gpu-venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/gpu/check_distributed.py --output runs/gpu-check/ddp.json
```

每张卡执行 BF16 矩阵乘法和反向传播，检查 NCCL all-reduce 的求和结果，再执行一次
DDP 优化器更新。不同 rank 使用不同目标，最后检查各卡权重一致且确实发生更新。
这能发现通信和梯度同步问题，但不是多卡训练吞吐测试。

## 图文推理与 LoRA

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  runs/gpu-venv/bin/python experiments/gpu/vlm_smoke.py \
  --model runs/models/Qwen3-VL-2B-Instruct \
  --output runs/gpu-check/vlm --steps 8
```

脚本画一张无答案标记的拓扑图，询问 `core` 和 `database` 链路的端点。
先保留原始模型回答，再用这两个样例跑 8 次 LoRA 更新；只监督回答 token。
使用 BF16、SDPA、梯度检查点和语言层 q/v 投影上的 rank-8 LoRA，micro-batch 为 1。

检查项包括有限 loss、非零有限梯度、adapter 权重变化、保存成功，以及重新加载后的
贪心输出与保存前一致。输出目录包含图、问题、逐步 loss、环境信息和 adapter。

这是可运行性检查。不能把这两个训练样例上的 loss 或回答，解释为故障定位能力、
训练收益或测试集准确率。正式实验还需要按拓扑划分训练／验证／测试集，固定预算并做对照。

## 结果记录

2026-10-02 在 8 张 RTX 5090、驱动 580.95.05 上实测：

| 检查 | 结果 |
| --- | --- |
| BF16 矩阵乘法与反向传播 | 8 张卡全部通过，构建包含 `sm_120` |
| NCCL all-reduce | 每张卡均得到预期的 36 |
| DDP 一次参数更新 | 各卡权重最大差为 0，参数确实改变 |
| Qwen3-VL 图文推理 | 两个原始回答为 `r1, r2` 和 `r2, db` |
| 单卡 LoRA | 完成 8 步，1,605,632 个可训练参数，梯度与 loss 均有限 |
| Adapter 重新加载 | 两个贪心输出均与保存前一致 |

该固定输入下，LoRA 检查的 PyTorch 峰值已分配显存为 4.57 GiB，
不包含其他进程和 CUDA 上下文开销，也不代表长序列或更大 batch 的显存需求。
原模型在训练前已经答对这两个问题，因此这里没有证明训练带来准确率提升。

[八卡结果](results/2026-10-02/ddp.json) · [LoRA 逐步结果](results/2026-10-02/vlm-smoke.json) ·
[输入图](results/2026-10-02/topology.png) · [问题与答案](results/2026-10-02/fixtures.json) ·
[源码哈希与运行记录](results/2026-10-02/provenance.json) · [依赖版本](requirements-linux-py312.lock)。

没有发布服务器地址、账号或连接凭据。所有检查使用新输出文件或目录，避免覆盖已有实验。
