#!/usr/bin/env python3
"""Upload an existing saved PEFT adapter + processor to the HF Model Hub.

This script never publishes the base model, datasets, cache, or training checkpoints.
It stages just the adapter dir plus the model card, so unrelated files aren't leaked.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-id', required=True, help='username/repo_name')
    parser.add_argument('--adapter-dir', default='nanovlm-medic-adapter')
    parser.add_argument('--card', default='HF_MODEL_CARD.md')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--yes', action='store_true', help='Actually upload; otherwise dry-run')
    args = parser.parse_args()

    source = Path(args.adapter_dir).expanduser().resolve()
    card = Path(args.card).resolve()
    if not source.is_dir():
        raise SystemExit(f'Adapter folder not found: {source}')
    if not (source / 'adapter_config.json').is_file():
        raise SystemExit('Missing adapter_config.json: was trainer.model.save_pretrained() run?')
    weights = list(source.glob('adapter_model.safetensors')) + list(source.glob('adapter_model.bin'))
    if not weights:
        raise SystemExit('Missing adapter_model.safetensors or adapter_model.bin')
    if not card.is_file():
        raise SystemExit(f'Model card missing: {card}')
    if args.repo_id.count('/') != 1 or args.repo_id.startswith('YOUR_'):
        raise SystemExit('Use --repo-id your_username/nanovlm-slake-500m-qlora')

    cfg = json.loads((source/'adapter_config.json').read_text())
    print('Adapter base model:', cfg.get('base_model_name_or_path'))
    print('Adapter folder:', source)
    print('Files selected for model repository:')
    for p in sorted(source.iterdir()):
        if p.is_file() and not p.name.startswith('.'):
            print(f'  {p.name}: {p.stat().st_size/1024**2:.2f} MiB')
    print('  README.md: model card')
    if not args.yes:
        print('DRY RUN: no upload performed. Add --yes to publish.')
        return

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.whoami()
    print('Authenticated on Hugging Face as:', info.get('name'))
    with tempfile.TemporaryDirectory(prefix='nanovlm_hf_release_') as tmp:
        stage = Path(tmp)
        for p in source.iterdir():
            if p.is_file() and not p.name.startswith('.') and p.name != 'README.md':
                shutil.copy2(p, stage/p.name)
        shutil.copy2(card, stage/'README.md')
        api.create_repo(repo_id=args.repo_id, repo_type='model', exist_ok=True, private=args.private)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type='model',
            folder_path=str(stage),
            commit_message='Publish NanoVLM-SLAKE QLoRA adapter and processor',
        )
    print(f'Success: https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
