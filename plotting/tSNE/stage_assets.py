#!/usr/bin/env python3
"""Inventory or copy the exact external files needed by a portable manifest.

Nothing is copied without --copy. This helper never writes weights or raw events
into Git. On the originating machine, --source-root names the directory relative
to which a private inventory locates files. Transfer the destination separately.
"""
import argparse
import json
from pathlib import Path
import shutil

from run import below, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True, help='Private local asset-location map; keep outside Git')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--copy', action='store_true', help='Actually copy; otherwise print the required files and sizes')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads(args.inventory.read_text())['assets']
    models = {m for g in manifest['groups'] for r in g['rows'].values() for m in r['models']}
    sources = {s for g in manifest['groups'] for r in g['rows'].values() for s in r['sources']}
    needed = {('sample',manifest['sources'][s]['path']) for s in sources}
    needed.update(('model',manifest['models'][m]['directory']+'/'+f) for m in models
                  for f in ('best_model.zip','input.json','trainset_stat.json'))
    selected = [a for a in inventory if (a['kind'],a['destination']) in needed]
    if {(a['kind'],a['destination']) for a in selected} != needed:
        raise ValueError('Inventory does not cover the selected manifest')
    # Preflight the complete list before creating any large asset copies.
    for a in selected:
        source = below(args.source_root,a['source'])
        if source.stat().st_size != a['bytes']:
            raise ValueError(f'Asset size changed: {source}')
        print(a['bytes'], a['kind'], a['destination'])
    print(f'{len(selected)} files; {sum(a["bytes"] for a in selected):,} bytes')
    if not args.copy:
        print('Dry run only. Add --copy to create the external asset directory.')
        return
    records = []
    for a in selected:
        source = below(args.source_root,a['source'])
        root = args.destination/('models' if a['kind']=='model' else 'samples')
        target = below(root,a['destination'])
        digest = sha256(source)
        if a.get('sha256',digest) != digest:
            raise ValueError(f'Asset hash changed: {source}')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if sha256(target) != digest:
                raise ValueError(f'Refusing to overwrite a different asset: {target}')
        else:
            pending = target.with_suffix(target.suffix+'.copying')
            shutil.copyfile(source,pending)
            if sha256(pending) != digest:
                raise ValueError(f'Copy verification failed: {target}')
            pending.replace(target)
        records.append(dict(kind=a['kind'],destination=a['destination'],sha256=digest,bytes=a['bytes']))
    write_json(args.destination/'TRANSFER.json',dict(files=records))
    print('Copied and SHA-256 verified all selected assets.')


if __name__ == '__main__':
    main()
