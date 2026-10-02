"""Download one pinned model from an official host and verify every file."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote, urlencode


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def matches(path, entry):
    return path.stat().st_size == entry['size'] and digest(path) == entry['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', choices=['modelscope', 'huggingface'], default='modelscope')
    args = parser.parse_args()
    manifest_path = Path(__file__).with_name('model-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if not shutil.which('curl'):
        parser.error('curl is required for resumable HTTPS downloads')
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        name = entry['path']
        if Path(name).name != name:
            raise ValueError('Only flat model files are supported')
        target = args.output / name
        if target.exists():
            if not matches(target, entry):
                raise ValueError(f'Existing file does not match the pinned manifest: {name}')
            print(f'Verified existing {name}', flush=True)
            continue
        if args.source == 'modelscope':
            query = urlencode({'Revision': manifest['modelscope_revision'], 'FilePath': name})
            url = f"https://modelscope.cn/api/v1/models/{manifest['model_id']}/repo?{query}"
        else:
            url = (f"https://huggingface.co/{manifest['model_id']}/resolve/"
                   f"{manifest['huggingface_revision']}/{quote(name)}")
        partial = target.with_name(target.name + '.part')
        print(f'Downloading {name} ({entry["size"]:,} bytes) from {args.source}', flush=True)
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '3', '--connect-timeout', '15', '--max-time', '1200',
                        '--continue-at', '-', '--output', str(partial), url], check=True)
        if not matches(partial, entry):
            raise ValueError(f'Size or SHA-256 mismatch: {name}; partial file retained for inspection')
        partial.rename(target)
        print(f'Verified {name}', flush=True)
    (args.output / 'model-manifest.json').write_bytes(manifest_path.read_bytes())
    print('All pinned model files verified.', flush=True)


if __name__ == '__main__':
    main()
