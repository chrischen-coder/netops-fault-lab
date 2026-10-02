"""Run local image inference, LoRA updates, and an adapter save/reload check.

This deliberately tiny fixture validates the training path. It is not an
evaluation of diagnosis accuracy or evidence of generalization.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def draw_fixture(path):
    image = Image.new('RGB', (640, 420), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    points = {'api': (65, 110), 'monitor': (65, 310), 'r1': (250, 210),
              'r2': (425, 110), 'db': (575, 110), 'cache': (425, 310)}
    links = [('access', 'api', 'r1'), ('monitor', 'monitor', 'r1'), ('core', 'r1', 'r2'),
             ('database', 'r2', 'db'), ('cache', 'r1', 'cache')]
    for label, source, target in links:
        x1, y1 = points[source]
        x2, y2 = points[target]
        draw.line((x1, y1, x2, y2), fill='#6d8a98', width=3)
        draw.text(((x1+x2)/2, (y1+y2)/2-20), label, font=font, fill='#173f4b', anchor='mm',
                  stroke_width=3, stroke_fill='white')
    for label, (x, y) in points.items():
        draw.ellipse((x-9, y-9, x+9, y+9), fill='#173f4b')
        draw.text((x, y+25), label, font=font, fill='#173f4b', anchor='mm')
    image.save(path)
    return image


def load_base(path):
    return Qwen3VLForConditionalGeneration.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.output.exists():
        parser.error('Use positive steps and a new output directory')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error('This check requires a CUDA GPU with BF16 support')
    manifest_bytes = (args.model / 'model-manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    for entry in manifest['files']:
        path = args.model / entry['path']
        if path.stat().st_size != entry['size']:
            parser.error(f'Model file size mismatch: {entry["path"]}; rerun download_model.py')
    args.output.mkdir(parents=True)
    torch.manual_seed(17)
    torch.set_num_threads(4)
    started = time.monotonic()
    image = draw_fixture(args.output / 'topology.png')
    fixtures = [
        {'question': 'Which two nodes are connected by the link labeled core? Reply only with the two node names, separated by a comma.', 'answer': 'r1, r2'},
        {'question': 'Which two nodes are connected by the link labeled database? Reply only with the two node names, separated by a comma.', 'answer': 'r2, db'},
    ]
    (args.output / 'fixtures.json').write_text(json.dumps(fixtures, indent=2) + '\n')
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prompts, batches = [], []
    for fixture in fixtures:
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                               {'type': 'text', 'text': fixture['question']}]}]
        prompt = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors='pt')
        answer = {'role': 'assistant', 'content': [{'type': 'text', 'text': fixture['answer']}]}
        full = processor.apply_chat_template(messages + [answer],
                                             tokenize=True, add_generation_prompt=False,
                                             return_dict=True, return_tensors='pt')
        length = prompt['input_ids'].shape[1]
        assert torch.equal(prompt['input_ids'], full['input_ids'][:, :length]), 'Prompt boundary differs'
        labels = full['input_ids'].clone()
        labels[:, :length] = -100
        assert (labels != -100).sum().item() > 0
        full['labels'] = labels
        prompts.append(prompt.to('cuda'))
        batches.append(full.to('cuda'))

    def generate(model):
        model.eval()
        outputs = []
        with torch.inference_mode():
            for prompt in prompts:
                generated = model.generate(**prompt, max_new_tokens=16, do_sample=False,
                                           temperature=None, top_p=None, top_k=None, use_cache=True)
                outputs.append(processor.decode(generated[0, prompt['input_ids'].shape[1]:],
                                                skip_special_tokens=True).strip())
        return outputs

    base = load_base(args.model)
    before = generate(base)
    print(json.dumps({'stage': 'inference', 'outputs': before}), flush=True)
    targets = [name for name, module in base.named_modules()
               if '.language_model.' in name and name.endswith(('.q_proj', '.v_proj'))]
    assert targets, 'No language attention projections found'
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                          target_modules=targets, bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-4)
    losses, grad_norms = [], []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batches[step % len(batches)], use_cache=False).loss
        assert torch.isfinite(loss), 'Non-finite loss'
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0, error_if_nonfinite=True)
        assert norm > 0, 'No adapter gradient'
        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(norm.item())
        print(json.dumps({'stage': 'train', 'step': step + 1, 'loss': losses[-1]}), flush=True)
    change = max((p.detach().cpu() - initial[name]).abs().max().item() for name, p in trainable.items())
    assert change > 0, 'Adapter did not update'
    after = generate(model)
    adapter = args.output / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    config_path = adapter / 'adapter_config.json'
    config = json.loads(config_path.read_text())
    config['base_model_name_or_path'] = manifest['model_id']
    config['revision'] = manifest['huggingface_revision']
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    trainable_count = sum(p.numel() for p in trainable.values())
    peak_memory = torch.cuda.max_memory_allocated() / 2**30
    del model, base, optimizer, trainable, initial, loss
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(args.model), adapter, local_files_only=True)
    restored = generate(reloaded)
    assert restored == after, 'Reloaded adapter generated different outputs'
    torch.cuda.synchronize()
    result = {
        'purpose': 'environment smoke test; two training fixtures, no held-out evaluation',
        'model_id': manifest['model_id'], 'model_revision': manifest['huggingface_revision'],
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'pillow']},
        'gpu': torch.cuda.get_device_name(), 'cuda_runtime': torch.version.cuda,
        'dtype': 'bfloat16', 'attention': 'sdpa', 'training_examples': len(fixtures),
        'optimizer_steps': args.steps, 'micro_batch_size': 1, 'learning_rate': 1e-4,
        'lora_rank': 8, 'lora_alpha': 16, 'trainable_parameters': trainable_count,
        'target_modules': targets, 'losses': losses, 'gradient_norms': grad_norms,
        'max_adapter_parameter_change': change, 'before': before, 'after': after,
        'adapter_reload_outputs': restored, 'adapter_reload_matches': True,
        'peak_training_allocated_gib': round(peak_memory, 3),
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'fixture_image_sha256': hashlib.sha256((args.output / 'topology.png').read_bytes()).hexdigest(),
    }
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'steps': args.steps, 'adapter_reload_matches': True}), flush=True)


if __name__ == '__main__':
    main()
