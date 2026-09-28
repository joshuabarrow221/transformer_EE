#!/usr/bin/env python3
"""Reproducible event t-SNE grids from saved predictions or latent NPZ files.

One point is one event. Labels are applied AFTER fitting; only the explicitly
selected numerical representation enters PCA/t-SNE. Read README.md before
interpreting cluster sizes, distances, or differences between model rows.
"""
from __future__ import annotations
import argparse
import gc
import fcntl
from contextlib import ExitStack
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
import sys
from pathlib import Path
import time

# Library progress messages must remain visible when this script is launched by
# the batch runner with stdout redirected to a log file.
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Keep caches outside a potentially read-only Git checkout. NUMBA_CPU_NAME is
# deliberately not forced: optional host workarounds belong in the environment.
_cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'transformeree-tsne'
for _name, _folder in [('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','matplotlib')]:
    os.environ.setdefault(_name,str(_cache/_folder))
    Path(os.environ[_name]).mkdir(parents=True,exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits
from topology import CATEGORIES, decode

TARGETS = ['Nu_Energy', 'Nu_Mom_X', 'Nu_Mom_Y', 'Nu_Mom_Z']
GENERATORS = ['AR23', 'G2111a', 'G1810a0211a', 'G1810a0211b', 'NuWro']
ROWS = ['Flat_Noise', 'Flat_NoNoise', 'Natural_Noise', 'Natural_NoNoise']
ROW_LABELS = ['Flat + noise', 'Flat + no noise', 'Natural + noise', 'Natural + no noise']
# Five hue families; shape and shade distinguish the paired proton counts.
COLORS = dict(zip(CATEGORIES, ['#B18422', '#2474A8', '#154A6C', '#D7792D',
                             '#914A16', '#BB628B', '#773651', '#77863A', '#AAAAAA']))
MARKERS = dict(zip(CATEGORIES, ['D', 'o', '^', 'o', '^', 'o', '^', 's', '.']))


def fingerprint(path):
    """Record file identity cheaply; selected data are additionally saved verbatim."""
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def sample_csv(entry, count, seed, policy='random'):
    """Close every streaming reader even when an alignment check fails."""
    with ExitStack() as stack:
        return _sample_csv(entry, count, seed, stack, policy)


def _sample_csv(entry, count, seed, stack, policy='random'):
    """Select either a uniform reservoir or the first valid file-order prefix.

    Sampling is label-blind and preserves the natural topology mixture. The
    same seed and row order give the same event indices across model outputs.
    Separate SV files are only composed when row counts and truth labels agree;
    this is a legacy row-order assumption, not a substitute for unique event IDs.
    With policy='first', only the necessary prefix is scanned; population counts
    then describe that scanned prefix, not the entire source file. Full-file
    population counts for the enlarged run are in sample_size_audit/.
    """
    paths = entry.get('paths', [entry.get('path')])
    readers = []
    for path in paths:
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in header if c in ['true_Topology', 'Event_Index']
                or c in [f'{prefix}_{t}' for prefix in ('true', 'pred') for t in TARGETS]]
        if 'true_Topology' not in cols:
            raise ValueError(f'Missing truth topology in {path}')
        readers.append(stack.enter_context(pd.read_csv(path, usecols=cols, dtype={'true_Topology': str}, chunksize=50000)))
    rng = np.random.default_rng(seed)
    kept = pd.DataFrame()
    counts, bad, total = Counter(), 0, 0
    from itertools import zip_longest
    for chunks in zip_longest(*readers):
        if any(c is None for c in chunks) or len({len(c) for c in chunks}) != 1:
            raise ValueError('SV component row counts differ')
        frame = chunks[0].copy()
        # Exact decimal comparison is safe for the 15-digit truth code.
        for component in chunks[1:]:
            # Integer and '.0' spellings denote the same exact decimal code.
            if not np.array_equal(frame.true_Topology.map(lambda v: decode(v)[0]),
                                  component.true_Topology.map(lambda v: decode(v)[0])):
                raise ValueError('SV component truth topologies do not align')
            for col in component:
                if col in frame and col != 'true_Topology':
                    if not np.allclose(frame[col], component[col], rtol=1e-6, atol=1e-8, equal_nan=True):
                        raise ValueError(f'SV component shared field does not align: {col}')
                elif col not in frame:
                    frame[col] = component[col].to_numpy()
        required = [f'pred_{t}' for t in TARGETS]
        if not all(c in frame for c in required):
            raise ValueError('All four predicted kinematic components are required')
        frame['source_row'] = np.arange(total, total+len(frame))
        frame['_priority'] = rng.random(len(frame))
        total += len(frame)
        decoded = []
        for value in frame.true_Topology:
            try:
                decoded.append(decode(value))
            except ValueError:
                decoded.append((None, -1, -1, 'Invalid'))
        frame[['topology_code', 'n_protons', 'n_pions', 'category']] = pd.DataFrame(decoded, index=frame.index)
        finite = np.isfinite(frame[required].to_numpy(float)).all(axis=1)
        valid = finite & (frame.category != 'Invalid')
        bad += int((~valid).sum())
        counts.update(frame.loc[valid, 'category'])
        kept = pd.concat([kept, frame.loc[valid]], ignore_index=True)
        if policy == 'first':
            # Invalid records are skipped; never duplicate events to fill a panel.
            kept = kept.head(count)
            if len(kept) == count:
                break
        else:
            kept = kept.nsmallest(count, '_priority')
    if policy == 'first' and len(kept) != count:
        raise ValueError(f'Required {count} valid events; found only {len(kept)}')
    if len(kept) < 40:
        raise ValueError(f'Only {len(kept)} usable events')
    return kept.sort_values('source_row').drop(columns='_priority').reset_index(drop=True), dict(
        total_rows=total, invalid_rows=bad, population_counts=dict(counts),
        sampling_policy=policy, counts_scope='scanned prefix' if policy=='first' else 'complete file',
        sources=[fingerprint(p) for p in paths],
        alignment='legacy row order + shared truth checks' if len(paths)>1 else 'single file')


def fit_embedding(features, seed, perplexity, iterations, pca_dimensions=50, backend='sklearn', threads=None, tsne_device=0, gpu_learning_rate=200.0):
    """Preserve Euclidean representation geometry; never standardize beam px/py.

    Per-component z-scoring would inflate tiny beam transverse predictions and
    can make numerical noise dominate. PCA is unwhitened and only reduces high
    dimensional latent vectors. Units of prediction components use c=1.
    """
    if backend not in ('fft', 'sklearn', 'cuml'):
        raise ValueError(f'Unknown t-SNE backend: {backend}')
    if backend == 'cuml':
        from gpu_tsne import runtime
        runtime(tsne_device)  # Fail before expensive CPU PCA if CUDA is unavailable.
    # Float32 input/PCA avoids several simultaneous multi-GB SV matrices.
    features = np.asarray(features, dtype=np.float32 if backend in ('fft','cuml') else np.float64)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite features')
    if len(features) <= perplexity:
        raise ValueError('Perplexity must be less than the number of sampled events')
    work = features
    info = dict(input_dimensions=features.shape[1], seed=seed, perplexity=perplexity,
                iterations=iterations, metric='euclidean', scaling='none', pca_whiten=False)
    if threads is None:
        threads=int(os.environ.get('TOPOLOGY_TSNE_THREADS','8')) if backend in ('fft','cuml') else 4
    if features.shape[1] > pca_dimensions:
        pca_start=time.perf_counter()
        print(f'PCA: {len(features):,} events × {features.shape[1]} features',flush=True)
        with threadpool_limits(limits=threads):
            pca = PCA(n_components=min(pca_dimensions, len(features)-1), random_state=seed)
            work = pca.fit_transform(features)
            info['pca_explained_variance'] = float(pca.explained_variance_ratio_.sum())
        info['pca_seconds']=time.perf_counter()-pca_start
        print(f"PCA complete in {info['pca_seconds']:.1f}s; variance {info['pca_explained_variance']:.6f}",flush=True)
    start = time.perf_counter()
    with threadpool_limits(limits=threads):
        if backend == 'cuml':
            from gpu_tsne import fit as fit_gpu
            coordinates, divergence, gpu_info = fit_gpu(work, seed, perplexity, iterations,
                device=tsne_device, learning_rate=gpu_learning_rate)
            info.update(gpu_info)
        elif backend == 'fft':
            import openTSNE
            # Construct the chosen index explicitly. The generic dispatcher
            # imports optional pynndescent even for exact/Annoy searches; that
            # triggers unnecessary LLVM startup on ARM. The same index classes,
            # seeds, k, and MultiscaleMixture produce identical P and coordinates
            # (see common_n199990/neighbor_construction_validation.json).
            neighbors='exact' if work.shape[1]<=4 else 'annoy'
            index_class=(openTSNE.nearest_neighbors.Sklearn if neighbors=='exact'
                         else openTSNE.nearest_neighbors.Annoy)
            k=min(len(work)-1,int(3*perplexity))
            index=index_class(data=work,k=k,metric='euclidean',n_jobs=threads,
                              random_state=seed,verbose=True)
            affinities=openTSNE.affinity.MultiscaleMixture(perplexities=[perplexity],
                knn_index=index,n_jobs=threads,random_state=seed,verbose=True)
            fit = openTSNE.TSNE(n_components=2, perplexity=perplexity,
                initialization='pca', learning_rate='auto', early_exaggeration_iter=250,
                n_iter=iterations-250, random_state=seed, negative_gradient_method='fft',
                neighbors=neighbors, n_jobs=threads, verbose=True)
            embedding = fit.fit(work,affinities=affinities)
            coordinates = np.asarray(embedding).copy()
            divergence = float(embedding.kl_divergence)
            info.update(backend='openTSNE FFT', backend_version=openTSNE.__version__,
                        threads=threads,
                        neighbors=neighbors,knn_k=k,affinity_construction='explicit index; standard MultiscaleMixture',
                        learning_rate_convention='openTSNE auto; differs from sklearn',
                        early_exaggeration_iterations=250)
        else:
            fit = TSNE(n_components=2, perplexity=perplexity, init='pca',
                   learning_rate='auto', max_iter=iterations, random_state=seed,
                   method='barnes_hut', n_jobs=4)
            coordinates = fit.fit_transform(work)
            divergence = float(fit.kl_divergence_)
        # Bounded diagnostic evaluated on a uniform subset; report its own n.
        ids = np.random.default_rng(seed).choice(len(work), min(1000, len(work)), replace=False)
        info['trustworthiness_subset_n'] = len(ids)
        info['trustworthiness_10nn'] = float(trustworthiness(features[ids], coordinates[ids], n_neighbors=10))
    info.update(kl_divergence=divergence, seconds=time.perf_counter()-start,
                sample_n=len(features), pca_dimensions=work.shape[1])
    if not np.isfinite(coordinates).all() or not np.isfinite(divergence):
        raise ValueError('t-SNE failed to produce finite coordinates')
    return coordinates, info


def draw_grid(group, frames, output, representation, coordinate_note, settings_note=None, style=None):
    """A shared legend and 4x5 layout keep the user's comparisons in fixed cells."""
    style = style or {}
    colors = {**COLORS, **style.get('colors', {})}
    fig, axes = plt.subplots(4, 5, figsize=(19, 14))
    fig.subplots_adjust(left=.09, right=.985, bottom=.10, top=.85, hspace=.19, wspace=.10)
    for i, row in enumerate(ROWS):
        # Prediction space has one fit for the entire grid. Latent fits are per
        # model row; generators in each row share coordinates and axis limits.
        row_frames = [f for (r, _), f in frames.items()
                      if r==row or representation=='prediction-space t-SNE']
        if row_frames:
            coords = pd.concat(row_frames)[['TSNE1','TSNE2']].to_numpy()
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            pad = np.maximum(hi-lo, 1)*.05
        for j, gen in enumerate(GENERATORS):
            ax = axes[i,j]
            if i==0: ax.set_title(gen, fontsize=13, pad=13)
            if j==0: ax.set_ylabel(ROW_LABELS[i], fontsize=12, labelpad=12)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color('#DDDDDD'); spine.set_linewidth(.5)
            frame = frames.get((row,gen))
            if frame is None:
                reason = group.get('missing', {}).get(f'{row}/{gen}', 'No matching source available')
                ax.text(.5,.5,reason,ha='center',va='center',transform=ax.transAxes,
                        fontsize=9,color='#666666',wrap=True)
                continue
            # Shuffle drawing order to avoid always covering one topology with another.
            f = frame.sample(frac=1, random_state=7)
            # One collection keeps all events in a genuinely interleaved order
            # without thousands of artists per panel. Each event retains its
            # category-specific marker as well as its explicitly mapped color.
            marker_paths={}
            for category in CATEGORIES:
                marker=MarkerStyle(MARKERS[category])
                marker_paths[category]=marker.get_path().transformed(marker.get_transform())
            collection=ax.scatter(f.TSNE1,f.TSNE2,c=f.category.map(colors).tolist(),
                                  s=style.get('point_size', .35 if len(f)>10000 else 4),
                                  alpha=style.get('alpha', .45 if len(f)>10000 else .7),
                                  linewidths=0,rasterized=True)
            collection.set_paths([marker_paths[c] for c in f.category])
            ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1])
            ax.set_aspect('equal',adjustable='box')
            ax.text(.02,.98,f'n = {len(frame):,}',transform=ax.transAxes,va='top',fontsize=8,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1.5))
    handles=[Line2D([],[],linestyle='',marker=MARKERS[c],color=colors[c],markersize=6,
                    label=c.replace('pi',r'$\pi$')) for c in CATEGORIES]
    fig.suptitle(group['title']+' — '+representation,fontsize=18,y=.975)
    training_note=(group['training_generator']+'-trained networks • ') if group.get('training_generator') else ''
    fig.text(.5,.939,training_note+group.get('inference_label','Natural-spectrum inference')+' • truth topology colors • '+coordinate_note,
             ha='center',fontsize=10)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=9,frameon=False)
    fig.text(.10,.047,'t-SNE 1 →       t-SNE 2 ↑',fontsize=10)
    # Keep a sensitivity figure identifiable after it is copied out of its
    # results folder, without changing any coordinates or plotted events.
    if settings_note:
        fig.text(.90,.047,settings_note,ha='right',fontsize=10)
    fig.text(.5,.025,'NpNπ: ≥3 protons OR ≥3 total pions. Other: 0p0π or 0p2π. '
             'π = π⁺ + π⁻ + π⁰. Labels never enter the embedding.',ha='center',fontsize=10)
    for ext in ('png','pdf'):
        # Write a complete sibling before replacement, preserving a valid old
        # artifact if a mounted-filesystem viewer has locked the destination.
        temporary=output.with_name(output.name+'.writing').with_suffix('.writing.'+ext)
        fig.savefig(temporary,dpi=style.get('dpi',180),facecolor='white')
        temporary.replace(output.with_suffix('.'+ext))
    plt.close(fig)


def run_group(group, args):
    out = args.output/group['id']; out.mkdir(parents=True,exist_ok=True)
    from render import file_hash
    frames, audits, feature_blocks = {}, {}, {}
    for entry in group['entries']:
        key=(entry['row'],entry['generator']); label='/'.join(key)
        try:
            if args.representation=='prediction':
                frame,audit=sample_csv(entry,args.events,args.seed,getattr(args,'sampling','random'))
                feature=frame[[f'pred_{t}' for t in TARGETS]].to_numpy(float)
            else:
                path=Path(entry['latent_path'])
                with np.load(path,allow_pickle=False) as data:
                    # Expanded five-generator grids can otherwise keep every
                    # row's SV activations in RAM at once. Lazy mode loads only
                    # identifiers/predictions here and reads activations per fit.
                    feature=path if getattr(args,'lazy_latent_load',False) else data['features']
                    frame=pd.DataFrame({
                        'source_row':data['source_row'], 'Event_Index':data['event_index'],
                        'true_Topology':data['topology'].astype(str)})
                    decoded=[decode(v) for v in frame.true_Topology]
                    frame[['topology_code','n_protons','n_pions','category']]=pd.DataFrame(decoded)
                    if 'prediction' in data:
                        for k,t in enumerate(TARGETS): frame['pred_'+t]=data['prediction'][:,k]
                audit=json.loads(path.with_suffix('.json').read_text())
            frames[key],audits[label],feature_blocks[key]=frame,audit,feature
            print(group['id'],label,len(frame),'sampled',flush=True)
            del feature  # The dictionary owns this array until its fit finishes.
        except (ValueError,FileNotFoundError,KeyError) as exc:
            if getattr(args,'strict',False):
                raise
            group.setdefault('missing',{})[label]=str(exc)[:110]
            audits[label]={'error':str(exc)}
    fits={}
    fit_groups=[list(frames)] if args.representation=='prediction' else [[k for k in frames if k[0]==row] for row in ROWS]
    for keys in fit_groups:
        if not keys: continue
        fit_name='all' if args.representation=='prediction' else keys[0][0]
        cache_path=out/f'fit_{fit_name}.json'
        signature=dict(seed=args.seed,perplexity=args.perplexity,iterations=args.iterations,
                       events=args.events,backend=getattr(args,'backend','sklearn'),
                       sampling=getattr(args,'sampling','random'),
                       sources={('/'.join(k)):audits['/'.join(k)] for k in keys})
        # A portable cache must also distinguish changed scientific code,
        # numerical libraries, and SV scaling definitions after migration.
        packages=['numpy','scipy','scikit-learn']
        if getattr(args,'backend','sklearn')=='fft': packages.extend(['openTSNE','annoy'])
        signature.update(renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         software={name:version(name) for name in packages},
                         sv_blocks=group.get('sv_block_sizes_by_row',{}).get(fit_name,group.get('sv_block_sizes')))
        if getattr(args,'backend','sklearn')=='cuml':
            from gpu_tsne import runtime
            signature['gpu_environment']=runtime(args.tsne_device)[2]
            signature['gpu_learning_rate']=args.gpu_learning_rate
            signature['gpu_adapter_sha256']=hashlib.sha256(Path(__file__).with_name('gpu_tsne.py').read_bytes()).hexdigest()
        # A completed row remains useful if a later row/render is interrupted.
        # Match source provenance and settings before accepting saved coordinates.
        if cache_path.exists():
            cached=json.loads(cache_path.read_text())
            files=[out/('_'.join(k)+'.csv.gz') for k in keys]
            if (cached.get('signature')==signature and all(p.exists() for p in files)
                    and all(cached.get('coordinate_sha256',{}).get(p.name)==file_hash(p) for p in files)):
                saved=[pd.read_csv(p,dtype={'topology_code':str,'true_Topology':str}) for p in files]
                if all(len(f)==args.events and np.array_equal(f.source_row,frames[k].source_row)
                       and np.isfinite(f[['TSNE1','TSNE2']]).all().all() for k,f in zip(keys,saved)):
                    for k,f in zip(keys,saved):
                        frames[k]=f
                        feature_blocks.pop(k,None)
                    fits[fit_name]=cached['fit']
                    print(group['id'],fit_name,'REUSE COMPLETE FIT',flush=True)
                    continue
        if args.representation=='latent' and getattr(args,'lazy_latent_load',False):
            # Allocate once and copy one source at a time. np.concatenate on a
            # list of loaded arrays would temporarily double the full row's RAM.
            width=sum(audits['/'.join(keys[0])]['block_sizes'])
            features=np.empty((sum(len(frames[k]) for k in keys),width),dtype=np.float32)
            offset=0
            for key in keys:
                with np.load(feature_blocks[key],allow_pickle=False) as data:
                    block=data['features']
                    if block.shape!=(len(frames[key]),width):
                        raise ValueError('Inconsistent latent feature dimensions')
                    features[offset:offset+len(block)]=block
                    offset+=len(block)
                    del block
        else:
            features=np.concatenate([feature_blocks[k] for k in keys])
        # SV latents are four separate networks. Equal total block variance
        # prevents arbitrary activation scale from choosing which network wins.
        # Independently fitted rows can use different network widths. Preserve
        # the legacy global block setting for direct historical manifests.
        blocks=group.get('sv_block_sizes_by_row',{}).get(fit_name,group.get('sv_block_sizes'))
        block_scaling=[]
        if args.representation=='latent' and blocks:
            offset=0
            for width in blocks:
                block=features[:,offset:offset+width]
                mean=block.mean(axis=0)
                block-=mean
                rms=max(float(np.sqrt(np.mean(np.sum(block**2,axis=1)))),1e-12)
                block/=rms
                block_scaling.append(dict(width=width,mean=mean.tolist(),rms_norm=rms))
                offset+=width
            if offset!=features.shape[1]: raise ValueError('SV block sizes do not match feature dimension')
            del block  # A slice otherwise keeps the full concatenated matrix alive.
        if getattr(args,'strict',False) and any(len(frames[k])!=args.events for k in keys):
            raise ValueError('A populated panel does not have the required event count')
        # A shared optional budget file lets an overnight batch use cores freed
        # by completed workers, without interrupting a fit already in progress.
        # It changes parallelism only; event selection and fit settings persist.
        budget_path=args.output.parent/'thread_budget.json'
        threads=None
        if budget_path.exists():
            threads=json.loads(budget_path.read_text()).get(args.representation)
            if threads is not None and not 1<=int(threads)<=os.cpu_count():
                raise ValueError('Thread budget exceeds the available CPU count')
        coordinates,info=fit_embedding(features,args.seed,args.perplexity,args.iterations,
                                      backend=getattr(args,'backend','sklearn'),threads=threads,
                                      tsne_device=getattr(args,'tsne_device',0),
                                      gpu_learning_rate=getattr(args,'gpu_learning_rate',200.0))
        if block_scaling:
            info['scaling']='each SV network block centered and divided by its RMS norm'
            info['block_scaling']=block_scaling
        fits[fit_name]=info
        offset=0
        for key in keys:
            frame=frames[key]; n=len(frame)
            frame[['TSNE1','TSNE2']]=coordinates[offset:offset+n]
            frame['generator']=key[1]; frame['training_condition']=key[0]
            frame.to_csv(out/('_'.join(key)+'.csv.gz'),index=False,
                         compression={'method':'gzip','compresslevel':1})
            offset+=n
        # All backends publish a completed, provenance-checked fit.
        pending=cache_path.with_suffix('.json.writing')
        pending.write_text(json.dumps(dict(signature=signature,fit=info,
            coordinate_sha256={('_'.join(k)+'.csv.gz'):file_hash(out/('_'.join(k)+'.csv.gz')) for k in keys}),indent=2))
        pending.replace(cache_path)
        # A completed row needs only its coordinates for rendering. Releasing
        # its high-dimensional blocks keeps subsequent SV fits within RAM.
        for key in keys:feature_blocks.pop(key,None)
        del features,coordinates
        gc.collect()
    if frames:
        draw_grid(group,frames,out/'grid',
                  'prediction-space t-SNE' if args.representation=='prediction' else 'event latent t-SNE',
                  'one joint fit across all cells' if args.representation=='prediction' else 'joint generators per row; rows have independent fits',
                  f'Sensitivity: seed {args.seed}; perplexity {args.perplexity:g}'
                  if args.seed!=42 or args.perplexity!=30 else None)
    # Coordinate hashes allow CPU-only redraws to verify transferred files
    # without reopening weights, raw CSVs or multi-GB latent arrays.
    from render import file_hash
    coordinate_files={('_'.join(key)+'.csv.gz'):dict(sha256=file_hash(out/('_'.join(key)+'.csv.gz')),
        rows=len(frame), row=key[0], generator=key[1]) for key,frame in frames.items()}
    metadata={'coordinate_files':coordinate_files,'group':group,'representation':args.representation,'audits':audits,'fits':fits,
              'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    counts=[]
    for (row,gen),frame in frames.items():
        audit=audits[f'{row}/{gen}']
        scope=audit.get('counts_scope','complete file' if 'population_counts' in audit else 'not counted')
        for category in CATEGORIES:
            counts.append(dict(row=row,generator=gen,category=category,
                               sample_count=int((frame.category==category).sum()),
                               population_count=audit.get('population_counts',{}).get(category) if scope=='complete file' else None,
                               scanned_prefix_count=audit.get('population_counts',{}).get(category) if scope=='scanned prefix' else None,
                               count_scope=scope))
    pd.DataFrame(counts).to_csv(out/'topology_counts.csv',index=False)
    # The batch monitor treats metadata.json as the grid's completion marker.
    # Publish it atomically, only after coordinates, figures and counts exist.
    pending=out/'metadata.json.writing'
    pending.write_text(json.dumps(metadata,indent=2))
    pending.replace(out/'metadata.json')
    print(group['id'],len(frames),'cells',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--representation',choices=['prediction','latent'],default='prediction')
    parser.add_argument('--events',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--perplexity',type=float,default=30)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--sampling',choices=['random','first'],default='random')
    parser.add_argument('--backend',choices=['sklearn','fft','cuml'],default='sklearn')
    parser.add_argument('--tsne-device',type=int,default=0,help='CUDA device index for cuML only')
    parser.add_argument('--gpu-learning-rate',type=float,default=200.0,help='Explicit cuML learning rate')
    parser.add_argument('--strict',action='store_true',help='Fail instead of silently omitting an available panel')
    parser.add_argument('--lazy-latent-load',action='store_true',
                        help='Load activation matrices one fit row at a time to bound RAM')
    args=parser.parse_args()
    if args.events<40 or args.iterations<300: parser.error('Use >=40 events and >=300 iterations')
    for group in json.loads(args.manifest.read_text())['groups']:
        if not args.only or group['id'] in args.only:
            directory=args.output/group['id'];directory.mkdir(parents=True,exist_ok=True)
            # Independent batch workers may reach the same grid. Serialize that
            # grid; saved-fit provenance checks then reuse an already finished fit.
            with (directory/'.run.lock').open('a') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                run_group(group,args)


if __name__=='__main__': main()
