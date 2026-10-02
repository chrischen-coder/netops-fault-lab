# GPU setup and training smoke checks

[中文安装说明](README.zh-CN.md)

This optional directory checks CUDA, distributed gradients, image inference and
LoRA save/reload. It does **not** evaluate network diagnosis accuracy. The core
CLI still has no GPU dependency.

Validated on 2026-10-02 with eight RTX 5090 GPUs, driver 580.95.05, Python 3.12.14,
PyTorch 2.8.0+cu128, Transformers 4.57.6 and PEFT 0.18.1:

- All eight GPUs passed BF16 forward/backward and NCCL all-reduce checks.
- One DDP optimizer step changed the weights; maximum cross-rank difference was zero.
- Qwen3-VL-2B-Instruct read two link endpoints from a generated topology image.
- One GPU completed eight LoRA steps with finite losses and nonzero finite gradients.
- Saving and reloading the adapter reproduced both greedy outputs.

The base model already answered both fixtures correctly. This is an environment
check, not evidence of improved accuracy or generalization. Peak PyTorch allocated
memory was 4.57 GiB for this fixed input; this excludes CUDA context and other
processes and does not estimate larger workloads.

## Reproduce

Use a new virtual environment with Python 3.12 and an NVIDIA driver supporting
CUDA 12.8. The commands below use `uv`, the
[official PyTorch CUDA index](https://pytorch.org/get-started/previous-versions/)
and the [Tsinghua PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/).

```bash
uv venv --python 3.12 runs/gpu-venv
uv pip install --python runs/gpu-venv/bin/python --link-mode copy \
  -r experiments/gpu/requirements-cuda.txt -c experiments/gpu/requirements-linux-py312.lock \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python runs/gpu-venv/bin/python --link-mode copy \
  -r experiments/gpu/requirements.txt -c experiments/gpu/requirements-linux-py312.lock \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python3 experiments/gpu/download_model.py --source modelscope \
  --output runs/models/Qwen3-VL-2B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  runs/gpu-venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/gpu/check_distributed.py --output runs/gpu-check/ddp.json
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  runs/gpu-venv/bin/python experiments/gpu/vlm_smoke.py \
  --model runs/models/Qwen3-VL-2B-Instruct --output runs/gpu-check/vlm --steps 8
```

Use idle GPUs; adjust visible devices and process count for smaller machines.
Model downloads require `curl`, support resume, and verify every file against the
[pinned manifest](model-manifest.json). `--source huggingface` downloads the same
revision from Hugging Face. The model uses Apache-2.0; weights are not bundled.
The checks run offline after download. Existing outputs are not overwritten.

[DDP results](results/2026-10-02/ddp.json) · [VLM results](results/2026-10-02/vlm-smoke.json) ·
[Fixture image](results/2026-10-02/topology.png) · [Fixture labels](results/2026-10-02/fixtures.json) ·
[Code hashes and provenance](results/2026-10-02/provenance.json).
