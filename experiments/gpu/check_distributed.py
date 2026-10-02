"""Check CUDA BF16, NCCL collectives and a DDP optimizer update on each GPU."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import platform

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new file')
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(17)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120), device_id=torch.device('cuda', rank))
    try:
        world = dist.get_world_size()
        matrix = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        value = (matrix @ matrix.T).float().square().mean()
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(matrix.grad).all()
        total = torch.tensor([dist.get_rank() + 1.0], device='cuda')
        dist.all_reduce(total)
        assert total.item() == world * (world + 1) / 2

        model = DistributedDataParallel(torch.nn.Linear(16, 4, bias=False).cuda(), device_ids=[rank])
        initial = model.module.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.ones(8, 16, device='cuda')
        target = torch.full((8, 4), (dist.get_rank() + 1) / world, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = (model(x).float() - target).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.module.weight.grad).all()
        optimizer.step()
        weights = [torch.empty_like(model.module.weight) for _ in range(world)]
        dist.all_gather(weights, model.module.weight.detach())
        mismatch = max((weights[0] - other).abs().max().item() for other in weights)
        change = (initial - model.module.weight).abs().max().item()
        assert mismatch < 1e-6 and change > 0
        torch.cuda.synchronize()
        device = torch.cuda.get_device_properties(rank)
        local = {'rank': dist.get_rank(), 'gpu': device.name, 'memory_gib': round(device.total_memory / 2**30, 3),
                 'capability': list(torch.cuda.get_device_capability(rank)), 'bf16_backward_finite': True,
                 'all_reduce_sum': total.item(), 'ddp_loss': loss.item(), 'max_weight_change': change,
                 'max_cross_rank_weight_difference': mismatch,
                 'peak_allocated_mib': round(torch.cuda.max_memory_allocated(rank) / 2**20, 2)}
        records = [None] * world
        dist.all_gather_object(records, local)
        if dist.get_rank() == 0:
            result = {'purpose': 'environment validation, not a training throughput benchmark',
                      'python': platform.python_version(), 'torch': torch.__version__,
                      'cuda_runtime': torch.version.cuda, 'nccl': list(torch.cuda.nccl.version()),
                      'world_size': world, 'architecture_builds': torch.cuda.get_arch_list(), 'gpus': records}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(result, stream, indent=2)
                stream.write('\n')
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
