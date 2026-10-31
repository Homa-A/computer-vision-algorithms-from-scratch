"""Content hashes: UTF8 text permits CRLF only; binaries remain byte-exact."""
import hashlib
import json
from pathlib import Path


def canonical_text(data):
    data = data.replace(b'\r\n', b'\n')
    if b'\r' in data:
        raise ValueError('Bare CR is not a line-ending equivalent.')
    text = data.decode('utf-8', errors='strict')
    if text.startswith('\ufeff'):
        raise ValueError('UTF8 BOM not permitted.')
    return data


def canonical(path):
    path = Path(path)
    data = path.read_bytes()
    return canonical_text(data) if path.suffix in ('.csv', '.json') else data


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def publish_manifest(directory):
    directory = Path(directory)
    entries = []
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.name == 'artifact_manifest.json':
            continue
        data = canonical(p)
        entries.append(dict(path=p.name, canonical_bytes=len(data), sha256=digest(data),
                            policy='utf8-crlf-to-lf' if p.suffix in ('.csv', '.json') else 'raw-binary'))
    write_json(directory/'artifact_manifest.json', {'files':entries})
