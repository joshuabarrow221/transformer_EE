#!/usr/bin/env python3
"""Portable, resumable TransformerEE latent extraction and topology t-SNE.

The manifest chooses model bundles and event sources explicitly. Extraction can
run on CUDA; optional cuML runs t-SNE on CUDA. PCA and rendering stay on CPU.
No training, normalization fitting, or label-based sampling takes place here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from topology import decode

HERE = Path(__file__).resolve().parent
TARGETS = ['Nu_Energy', 'Nu_Mom_X', 'Nu_Mom_Y', 'Nu_Mom_Z']
ROWS = ['Flat_Noise', 'Flat_NoNoise', 'Natural_Noise', 'Natural_NoNoise']
GENERATORS = ['AR23', 'G2111a', 'G1810a0211a', 'G1810a0211b', 'NuWro']


def sha256(path):
    """Hash large files incrementally without allocating their full contents."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """Publish complete records atomically, even with multiple shard workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        pending = Path(f.name)
    pending.replace(path)


@contextmanager
def lock(path):
    """Linux/WSL advisory locks serialize shared selections and duplicate tasks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Unsafe or empty identifier: {value!r}')
    return value


def below(root, relative):
    """Manifest assets are relative to explicit roots; reject accidental escapes."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset path escapes its root: {relative}')
    return path


def resolve(args):
    """Validate pairing, configuration, normalization, and checkpoint identity.

    Model IDs identify actual checkpoints, not a loss label or file timestamp.
    Four SV components may be combined only when each physical target occurs
    exactly once. The same bundle is used for all generators in a plotted row.
    """
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('version') != 1:
        raise ValueError('Expected manifest version 1')
    groups = manifest['groups']
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate group IDs')
    if args.only:
        unknown = set(args.only) - {g['id'] for g in groups}
        if unknown:
            raise ValueError(f'Unknown groups: {sorted(unknown)}')
        groups = [g for g in groups if g['id'] in args.only]
    if not groups:
        raise ValueError('No groups selected')
    model_ids, source_ids, tasks = set(), set(), []
    for group in groups:
        safe_id(group['id'])
        for row, pairing in group['rows'].items():
            if row not in ROWS or not pairing['models'] or not pairing['sources']:
                raise ValueError(f'Invalid or empty pairing: {group["id"]}/{row}')
            gens = [manifest['sources'][s]['generator'] for s in pairing['sources']]
            if len(set(gens)) != len(gens) or not set(gens) <= set(GENERATORS):
                raise ValueError('One source per supported generator is required in a row')
            model_ids.update(pairing['models'])
            source_ids.update(pairing['sources'])
            for source in pairing['sources']:
                safe_id(source)
                tasks.append(dict(id=f'{group["id"]}_{row}_{source}', group=group['id'],
                                  row=row, source=source, models=pairing['models']))
    models = {}
    for key in sorted(model_ids):
        item = manifest['models'][key]
        directory = below(args.models_root, item['directory'])
        hashes = {name: sha256(directory/name) for name in
                  ('best_model.zip', 'input.json', 'trainset_stat.json')}
        for name, expected in item.get('sha256', {}).items():
            if hashes.get(name) != expected:
                raise ValueError(f'Checkpoint bundle hash mismatch: {key}/{name}')
        config = json.loads((directory/'input.json').read_text())
        stats = json.loads((directory/'trainset_stat.json').read_text())
        if config['model']['name'] != 'Transformer_EE_MV':
            raise ValueError(f'Unsupported architecture: {key}')
        if 'Topology' in config['vector'] + config['scalar']:
            raise ValueError('Truth topology must not enter the model inputs')
        if not config['vector'] or not config['scalar'] or config['max_num_prongs'] < 1:
            raise ValueError('This exporter requires vector and scalar model inputs')
        for field in config['vector'] + config['scalar']:
            values = stats[field]
            if len(values) != 2 or not np.isfinite(values).all() or values[1] <= 0:
                raise ValueError(f'Invalid training normalization: {key}/{field}')
        if 'Topology' in config['target']:
            coefficients = config['loss']['kwargs']['coefficients']
            if coefficients[config['target'].index('Topology')] != 0:
                raise ValueError(f'Topology loss is active in {key}; this study requires zero')
        models[key] = dict(directory=str(directory), sha256=hashes, config=config,
                           trainset_statistics=stats)
    for task in tasks:
        targets = [t for m in task['models'] for t in models[m]['config']['target'] if t in TARGETS]
        if Counter(targets) != Counter(TARGETS):
            raise ValueError(f'Bundle must predict each physical target once: {task["id"]}')
    sources = {}
    for key in sorted(source_ids):
        item = manifest['sources'][key]
        path = below(args.samples_root, item['path'])
        stat = path.stat()
        sources[key] = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                            generator=item['generator'])
    # Check implementation files as well as checkpoint data for safe resume.
    code = {str(p.relative_to(args.repo)): sha256(p) for p in [
        args.repo/'transformer_ee/model/transformerEncoder.py',
        args.repo/'transformer_ee/dataloader/pd_dataset.py',
        args.repo/'transformer_ee/utils/weights.py']}
    code.update({p.name: sha256(p) for p in (HERE/'run.py', HERE/'inference.py', HERE/'topology.py')})
    return dict(groups=groups, tasks=tasks, models=models, sources=sources,
                code=code, events=args.events, selection='first usable events in CSV order')


def select_source(source, models, n):
    """Read only the necessary prefix using the union of selected model inputs.

    Invalid records are counted, then replaced by later usable records. Selection
    never depends on predicted errors or topology category. All paired networks
    receive precisely the same ordered events. Counts describe a scanned prefix,
    not the full generator population.
    """
    vectors = sorted({v for m in models.values() for v in m['config']['vector']})
    scalars = sorted({s for m in models.values() for s in m['config']['scalar']})
    required = set(vectors + scalars + TARGETS + ['Topology', 'Event_Index', 'Final_State_Particles_PDG'])
    kept, used, scanned, excluded = [], 0, 0, 0
    with pd.read_csv(source['path'], dtype={'Topology': str}, float_precision='round_trip', chunksize=8192) as reader:
        for chunk in reader:
            if required - set(chunk):
                raise ValueError(f'Missing input columns: {sorted(required-set(chunk))}')
            chunk['source_row'] = np.arange(scanned, scanned+len(chunk))
            scanned += len(chunk)
            numeric = chunk[scalars+TARGETS+['Event_Index']].apply(pd.to_numeric, errors='coerce')
            valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
            ids = numeric.Event_Index.to_numpy(float)
            valid &= (ids == np.floor(ids)) & (np.abs(ids) <= 2**53)
            # Parse strictly; np.fromstring alone can silently accept malformed
            # suffixes. Comparing lengths also catches misaligned prong vectors.
            for i in np.flatnonzero(valid):
                row = chunk.iloc[i]
                try:
                    arrays = [np.array([float(x) for x in row[v].split(',')]) for v in vectors]
                    if len({len(a) for a in arrays}) != 1 or any(not np.isfinite(a).all() for a in arrays):
                        raise ValueError('Invalid vectors')
                    pdgs = [float(x) for x in row.Final_State_Particles_PDG.split(',')]
                    if any(not np.isfinite(x) or x != int(x) for x in pdgs):
                        raise ValueError('Invalid PDGs')
                    _, p, pi, _ = decode(row.Topology)
                    if (pdgs.count(2212), sum(pdgs.count(x) for x in (211, -211, 111))) != (p, pi):
                        raise ValueError('Topology/PDG mismatch')
                except (ValueError, TypeError, AttributeError, OverflowError):
                    valid[i] = False
            excluded += int((~valid).sum())
            selected = chunk.loc[valid].iloc[:n-used]
            kept.append(selected)
            used += len(selected)
            if used == n:
                break
    if used != n:
        raise ValueError(f'Only {used} usable events; requested {n}: {source["path"]}')
    frame = pd.concat(kept, ignore_index=True)
    frame[scalars+TARGETS] = frame[scalars+TARGETS].apply(pd.to_numeric)
    frame['Event_Index'] = pd.to_numeric(frame.Event_Index).astype(np.int64)
    if not frame.Event_Index.is_unique:
        raise ValueError('Duplicate selected Event_Index values')
    return frame, dict(sampled_rows=n, scanned_rows=scanned, invalid_in_scanned_prefix=excluded,
                       excluded_before_last_selected=int(frame.source_row.iloc[-1])+1-n,
                       counts_scope='scanned prefix', last_source_row=int(frame.source_row.iloc[-1]))


def selected_frame(args, plan, key):
    """Cache one validated event cohort per source, shared by all model rows."""
    folder = args.output/'selected'
    with lock(folder/(key+'.lock')):
        csv, meta = folder/(key+'.csv.gz'), folder/(key+'.json')
        if csv.exists() and meta.exists():
            audit = json.loads(meta.read_text())
            if sha256(csv) != audit['csv_sha256']:
                raise ValueError(f'Selected cohort was modified: {csv}')
            frame = pd.read_csv(csv, dtype={'Topology': str}, float_precision='round_trip')
        else:
            frame, audit = select_source(plan['sources'][key], plan['models'], args.events)
            pending = folder/(key+'.writing.csv.gz')
            frame.to_csv(pending, index=False, compression='gzip')
            pending.replace(csv)
            audit['csv_sha256'] = sha256(csv)
            write_json(meta, audit)
        return frame, audit


def extract_task(args, plan, task, frame, audit):
    """Bound host/device memory by chunking and disk-backed output arrays.

    Each checkpoint stays on the selected device across chunks. Only normalized
    minibatches enter the GPU. FP32, eval(), and inference_mode() are retained;
    no autocast or TF32 optimization is introduced into this scientific export.
    """
    import torch
    from inference import load_network, forward_fast
    destination = args.output/'features'/(task['id']+'.npz')
    meta_path = destination.with_suffix('.json')
    with lock(destination.with_suffix('.lock')):
        if destination.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['npz_sha256'] != sha256(destination):
                raise ValueError(f'Feature export was modified: {destination}')
            print('REUSE', task['id'], flush=True)
            return
        n, blocks, pens, records = len(frame), [], [], []
        with tempfile.TemporaryDirectory(prefix=task['id']+'.', dir=destination.parent) as work, ExitStack() as mappings:
            work = Path(work)
            def mapped(name, shape):
                # Explicitly close mmap handles before TemporaryDirectory tries
                # to unlink them. Open mapped files block deletion on NTFS/WSL.
                array = np.lib.format.open_memmap(work/name, mode='w+', dtype='float32', shape=shape)
                def close():
                    array.flush()
                    array._mmap.close()
                mappings.callback(close)
                return array
            prediction = mapped('prediction.npy', (n,4))
            prediction[:] = np.nan
            for k, model_id in enumerate(task['models']):
                model = plan['models'][model_id]
                net, config, stats = load_network(args.repo, model['directory'], args.device)
                checks = []
                for start in range(0, n, args.chunk_size):
                    stop = min(start+args.chunk_size, n)
                    head, pen, pred, qa = forward_fast(net, config, stats,
                        frame.iloc[start:stop].reset_index(drop=True), args.device, args.batch_size,args.prepare_engine)
                    if start == 0:
                        h = mapped(f'head{k}.npy', (n,head.shape[1]))
                        p = mapped(f'pen{k}.npy', (n,pen.shape[1]))
                        blocks.append(h); pens.append(p)
                    h[start:stop], p[start:stop] = head, pen
                    for j, target in enumerate(config['target']):
                        if target in TARGETS:
                            prediction[start:stop,TARGETS.index(target)] = pred[:,j]
                    checks.append(qa)
                    print(f'{task["id"]} model {k+1}/{len(task["models"])}: {stop:,}/{n:,}', flush=True)
                records.append(dict(id=model_id, **model, forward_checks=checks))
                del net
            def combine(arrays, name):
                # Copy block by block into a memmap; never concatenate every
                # generator/model's high-dimensional arrays in host RAM.
                result = mapped(name, (n,sum(a.shape[1] for a in arrays)))
                offset = 0
                for a in arrays:
                    result[:,offset:offset+a.shape[1]] = a
                    offset += a.shape[1]
                return result
            features, penultimate = combine(blocks,'features.npy'), combine(pens,'penultimate.npy')
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite or missing physical predictions')
            pending = destination.with_suffix('.writing.npz')
            np.savez_compressed(pending, features=features, penultimate=penultimate, prediction=prediction,
                topology=frame.Topology.astype(str).to_numpy(dtype=str),
                event_index=frame.Event_Index.to_numpy(np.int64), source_row=frame.source_row.to_numpy(np.int64),
                truth=frame[TARGETS].to_numpy(float), block_sizes=np.array([a.shape[1] for a in blocks]))
            pending.replace(destination)
            write_json(meta_path, dict(task=task, models=records, block_sizes=[a.shape[1] for a in blocks],
                sample_audit=audit, feature='input to regression head', evaluation_noise=False,
                device=args.device, torch_version=torch.__version__, batch_size=args.batch_size,
                chunk_size=args.chunk_size, prepare_engine=args.prepare_engine, npz_sha256=sha256(destination)))


def plot(args, plan):
    """Feed the existing scientific renderer a resolved manifest of activations."""
    groups = []
    for original in plan['groups']:
        group = {k:v for k,v in original.items() if k != 'rows'}
        group['entries'] = []
        widths = {}
        for task in plan['tasks']:
            if task['group'] != group['id']:
                continue
            path = args.output/'features'/(task['id']+'.npz')
            meta = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != meta['npz_sha256']:
                raise ValueError(f'Changed activation file: {path}')
            with np.load(path, allow_pickle=False) as data:
                if len(data['source_row']) != args.events or not np.isfinite(data['prediction']).all():
                    raise ValueError(f'Invalid activation file: {path}')
            row_widths = widths.setdefault(task['row'],meta['block_sizes'])
            if row_widths != meta['block_sizes']:
                raise ValueError('Generators within one row must have identical feature block dimensions')
            group['entries'].append(dict(row=task['row'], generator=plan['sources'][task['source']]['generator'], latent_path=str(path)))
        group['sv_block_sizes_by_row'] = {row:sizes for row,sizes in widths.items() if len(sizes)>1}
        groups.append(group)
    manifest = args.output/'latent_manifest.json'
    write_json(manifest, dict(groups=groups))
    env = dict(os.environ, TOPOLOGY_TSNE_THREADS=str(args.threads))
    subprocess.run([sys.executable, str(HERE/'plot_tsne.py'), '--manifest', str(manifest),
        '--output', str(args.output/'plots'), '--representation', 'latent', '--events', str(args.events),
        '--seed', str(args.seed), '--perplexity', str(args.perplexity), '--iterations', str(args.iterations),
        '--backend', args.backend, '--tsne-device',str(args.tsne_device),
        '--gpu-learning-rate',str(args.gpu_learning_rate), '--sampling', 'first', '--strict', '--lazy-latent-load'], env=env, check=True)
    lines = ['# TransformerEE latent t-SNE', '', f'{args.events:,} events per populated panel.', '',
             '| Grid | PNG | PDF |', '|---|---|---|']
    for group in groups:
        stem = 'plots/'+group['id']+'/grid'
        lines.append(f'| {group["title"]} | [PNG]({stem}.png) | [PDF]({stem}.pdf) |')
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, help='Required for extraction/check; plot uses plan.json and render uses coordinates')
    p.add_argument('--repo', type=Path, default=HERE.parents[1])
    p.add_argument('--models-root', type=Path)
    p.add_argument('--samples-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['check','extract','plot','render','all'], default='all')
    p.add_argument('--device', default='cpu', help='cpu or cuda[:index]; never silently falls back')
    p.add_argument('--events', type=int, default=199990)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--chunk-size', type=int, default=8192)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--prepare-engine', choices=['pandas','polars'],default='pandas',
                   help='Vector parsing/normalization backend; selection and CSV audit stay shared')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--only', nargs='+', help='Group IDs; choose before starting a new output directory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--perplexity', type=float, default=30)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--backend', choices=['fft','sklearn','cuml'], default='fft')
    p.add_argument('--tsne-device',type=int,default=0,help='CUDA index for cuML, independent of inference --device')
    p.add_argument('--gpu-learning-rate',type=float,default=200.0)
    p.add_argument('--render-output',type=Path,help='Redrawn figures directory; defaults to OUTPUT/redraw')
    p.add_argument('--style',type=Path,help='Render-only JSON colors, title, alpha, point_size, dpi')
    args = p.parse_args()
    if args.stage == 'render':
        # This route must not resolve models or read activations/plan.json.
        from render import render_all
        render_all(args.output/'plots', args.render_output or args.output/'redraw', args.only, args.style)
        return
    if args.style or args.render_output:
        p.error('--style and --render-output are only valid with --stage render')
    if args.backend == 'cuml' and args.stage in ('check','plot','all'):
        from gpu_tsne import runtime, parameters
        runtime(args.tsne_device)
        parameters(args.events, args.seed, args.perplexity, args.iterations, args.gpu_learning_rate)
    os.environ.setdefault('POLARS_MAX_THREADS',str(args.threads))
    # Repository imports can initialize Matplotlib even during extraction.
    # Give that initialization the same persistent writable cache as plotting.
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'transformeree-tsne'
    os.environ.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
    for name in ['manifest','repo','models_root','samples_root','output']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if min(args.events,args.batch_size,args.chunk_size,args.threads,args.workers) < 1 or args.events < 40:
        p.error('Positive sizes and at least 40 events are required')
    if not 0 <= args.worker_index < args.workers or args.iterations < 300 or not 0 < args.perplexity < args.events:
        p.error('Invalid worker index, iterations, or perplexity')
    if args.workers > 1 and args.stage != 'extract':
        p.error('Sharding is supported only for extraction; plot once after all workers finish')
    if not re.fullmatch(r'cpu|cuda(?::\d+)?', args.device):
        p.error('Device must be cpu or cuda[:index]')
    if args.stage == 'plot':
        # Saved feature exports are sufficient to render on a different host;
        # original weights, raw CSVs, and their recorded absolute paths need not
        # exist there. Rebuild activation paths under the supplied output root.
        plan = json.loads((args.output/'plan.json').read_text())
        args.events = plan['events']
    else:
        if any(getattr(args, name) is None for name in ('manifest','models_root','samples_root')):
            p.error('--manifest, --models-root, and --samples-root are required for extraction/check')
        plan = resolve(args)
    print(f'Validated {len(plan["models"])} models, {len(plan["sources"])} sources, {len(plan["tasks"])} exports', flush=True)
    if args.stage in ('check','extract','all'):
        import torch
        if args.prepare_engine == 'polars':
            try:
                import polars
            except ImportError as exc:
                raise RuntimeError('Install requirements-polars.txt to use --prepare-engine polars') from exc
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; install the correct PyTorch build or use --device cpu')
            torch.cuda.get_device_properties(torch.device(args.device))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(args.threads)
    if args.stage == 'check':
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Extraction settings are audit metadata, not new physics. Pin the device
    # type/torch version during extraction; plots can run on another CPU host.
    if args.stage != 'plot':
        with lock(args.output/'.plan.lock'):
            path = args.output/'plan.json'
            if path.exists() and json.loads(path.read_text()) != plan:
                raise ValueError('Inputs, code, model selection, or event count changed; use a new --output directory')
            write_json(path, plan)
    if args.stage in ('extract','all'):
        with lock(args.output/'.extraction.lock'):
            environment = dict(device_type=args.device.split(':')[0], torch_version=torch.__version__,
                               batch_size=args.batch_size, chunk_size=args.chunk_size,
                               prepare_engine=args.prepare_engine,
                               polars_version=polars.__version__ if args.prepare_engine=='polars' else None)
            path = args.output/'extraction_environment.json'
            if path.exists() and json.loads(path.read_text()) != environment:
                raise ValueError('Extraction environment changed; use a new output directory')
            write_json(path, environment)
        previous, frame, audit = None, None, None
        for i, task in enumerate(sorted(plan['tasks'], key=lambda t:(t['source'],t['id']))):
            if i % args.workers != args.worker_index:
                continue
            if previous != task['source']:
                frame, audit = selected_frame(args, plan, task['source'])
                previous = task['source']
            extract_task(args, plan, task, frame, audit)
    if args.stage in ('plot','all'):
        # A distinct settings directory prevents mixing coordinate systems.
        settings = dict(seed=args.seed, perplexity=args.perplexity, iterations=args.iterations, backend=args.backend)
        if args.backend == 'cuml':
            settings.update(tsne_device=args.tsne_device,gpu_learning_rate=args.gpu_learning_rate)
        with lock(args.output/'.plot.lock'):
            path = args.output/'plot_settings.json'
            if path.exists() and json.loads(path.read_text()) != settings:
                raise ValueError('Plot settings changed; use plot_tsne.py with a separate --output for sensitivity fits')
            write_json(path, settings)
            plot(args, plan)


if __name__ == '__main__':
    main()
