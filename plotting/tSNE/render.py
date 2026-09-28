#!/usr/bin/env python3
"""Redraw existing t-SNE coordinates without inference, PCA, or fitting.

Only completed grid metadata and coordinate CSV.gz files are required. Write
presentation changes to a separate directory so original fit artifacts retain
provenance. Historical exports lacking hashes are supported with a warning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from topology import CATEGORIES, decode


def file_hash(path):
    """Hash in bounded chunks; never load a whole compressed file just to hash."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_style(path):
    """Keep presentation overrides explicit and independent of scientific fits."""
    style = json.loads(Path(path).read_text()) if path else {}
    allowed = {'colors', 'title', 'point_size', 'alpha', 'dpi'}
    if not isinstance(style, dict) or set(style) - allowed:
        raise ValueError(f'Style must contain only {sorted(allowed)}')
    from matplotlib.colors import is_color_like
    colors = style.get('colors', {})
    if not isinstance(colors, dict) or set(colors) - set(CATEGORIES) or not all(is_color_like(c) for c in colors.values()):
        raise ValueError('Style colors must map known topology categories to valid colors')
    for key in ['point_size', 'alpha', 'dpi']:
        if key in style and (not isinstance(style[key], (int, float)) or not np.isfinite(style[key]) or style[key] <= 0):
            raise ValueError(f'Style {key} must be finite and positive')
    if style.get('alpha', 1) > 1 or style.get('dpi', 180) > 1200:
        raise ValueError('Require alpha <= 1 and dpi <= 1200')
    if 'title' in style and not isinstance(style['title'], str):
        raise ValueError('Style title must be a string')
    return style


def load_coordinates(directory):
    """Verify identifiers, labels, counts and (for new exports) file integrity."""
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    records = metadata.get('coordinate_files')
    if records is None:
        warnings.warn(f'{directory.name}: historical coordinates have no SHA-256 inventory; '
                      'checking counts, labels and identifiers only')
        records = {}
        for entry in metadata['group']['entries']:
            row, gen = entry['row'], entry['generator']
            if 'error' in metadata.get('audits', {}).get(f'{row}/{gen}', {}):
                continue
            records[f'{row}_{gen}.csv.gz'] = dict(row=row, generator=gen)
    if not records:
        raise ValueError('No completed coordinate panels in metadata')
    frames = {}
    for filename, record in records.items():
        if Path(filename).name != filename:
            raise ValueError('Coordinate filenames must stay within their grid directory')
        path = directory/filename
        if 'sha256' in record and file_hash(path) != record['sha256']:
            raise ValueError(f'Coordinate hash mismatch: {filename}')
        frame = pd.read_csv(path, dtype={'topology_code':str, 'true_Topology':str})
        key = (record['row'], record['generator'])
        required = {'source_row','true_Topology','category','TSNE1','TSNE2','generator','training_condition'}
        if not required <= set(frame):
            raise ValueError(f'Missing coordinate columns: {filename}')
        expected = record.get('rows')
        if expected is not None and len(frame) != expected:
            raise ValueError(f'Coordinate count mismatch: {filename}')
        if len(frame) == 0 or frame.source_row.isna().any() or frame.source_row.duplicated().any():
            raise ValueError(f'Invalid event identifiers: {filename}')
        if not np.isfinite(frame[['TSNE1','TSNE2']].to_numpy(float)).all():
            raise ValueError(f'Nonfinite coordinates: {filename}')
        if not (frame.generator.eq(key[1]).all() and frame.training_condition.eq(key[0]).all()):
            raise ValueError(f'Coordinate panel identity mismatch: {filename}')
        if not frame.category.eq(frame.true_Topology.map(lambda v: decode(v)[3])).all():
            raise ValueError(f'Truth topology labels disagree: {filename}')
        if key in frames:
            raise ValueError(f'Duplicate coordinate panel: {key}')
        frames[key] = frame
    # Fits report the total joint sample count, including all generator panels.
    for name, fit in metadata['fits'].items():
        count = sum(len(f) for (row, _), f in frames.items() if name == 'all' or row == name)
        if count != fit['sample_n']:
            raise ValueError(f'Joint-fit event count mismatch: {name}')
    return metadata, frames


def render_all(source, output, only=None, style_path=None):
    """Read each grid independently to bound host memory; never import CUDA."""
    # Lazy import avoids a circular dependency when the fitter imports file_hash.
    from plot_tsne import draw_grid
    source, output = Path(source).resolve(), Path(output).resolve()
    style = read_style(style_path)
    paths = [source/'metadata.json'] if (source/'metadata.json').exists() else sorted(source.glob('*/metadata.json'))
    found = set()
    for path in paths:
        group_id = path.parent.name
        if only and group_id not in only:
            continue
        found.add(group_id)
        destination = output/group_id
        if destination.resolve() == path.parent.resolve():
            raise ValueError('Render output must be separate from the original fit directory')
        metadata, frames = load_coordinates(path.parent)
        group = dict(metadata['group'])
        if 'title' in style:
            group['title'] = style['title']
        destination.mkdir(parents=True, exist_ok=True)
        prediction = metadata['representation'] == 'prediction'
        args = metadata['arguments']
        draw_grid(group, frames, destination/'grid',
            'prediction-space t-SNE' if prediction else 'event latent t-SNE',
            'one joint fit across all cells' if prediction else 'joint generators per row; rows have independent fits',
            f"Saved coordinates: seed {args['seed']}; perplexity {args['perplexity']:g}", style=style)
        audit = dict(source_metadata_sha256=file_hash(path), style=style,
                     coordinate_files={f'{r}_{g}.csv.gz':file_hash(path.parent/f'{r}_{g}.csv.gz') for r,g in frames},
                     operation='render only; coordinates unchanged')
        pending = destination/'render.json.writing'
        pending.write_text(json.dumps(audit, indent=2))
        pending.replace(destination/'render.json')
        print(f'{group_id}: redrawn {sum(map(len, frames.values())):,} saved events; no fit', flush=True)
    if not found or (only and set(only)-found):
        raise ValueError(f'No completed metadata for requested grids: {set(only or [])-found}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='plots directory or one completed grid directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--style',type=Path)
    args = parser.parse_args()
    render_all(args.source, args.output, args.only, args.style)


if __name__ == '__main__':
    main()
