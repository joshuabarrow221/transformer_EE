#!/usr/bin/env python3
"""Export event activations from real TransformerEE checkpoints on CPU or GPU.

This module imports the repository's model and normalization code. Hooks observe
the original forward pass rather than reimplementing pooling or changing model
behavior. Checkpoints, inputs, and training statistics remain untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from topology import decode


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_network(repo, directory, device='cpu'):
    """Require architecture, checkpoint and training normalization as a bundle."""
    sys.path.insert(0,str(Path(repo).resolve()))
    from transformer_ee.model.transformerEncoder import Transformer_EE_MV
    directory=Path(directory)
    config=json.loads((directory/'input.json').read_text())
    stats=json.loads((directory/'trainset_stat.json').read_text())
    if config['model']['name']!='Transformer_EE_MV':raise ValueError('Unsupported architecture')
    # Refuse to accidentally include a truth label in the observed inputs.
    if 'Topology' in config['vector']+config['scalar']:
        raise ValueError('Truth topology is an input feature in this configuration')
    net=Transformer_EE_MV(config)
    net.load_state_dict(torch.load(directory/'best_model.zip',map_location='cpu',weights_only=True),strict=True)
    return net.to(device).eval(),config,stats


def forward_frame(net, config, stats, frame, device='cpu', batch_size=64):
    """Return head input, penultimate activations, predictions, and QA counts.

    Legacy head input includes the masked-sum pooled encoder and learned scalar
    branch. For scalar_as_token/new_head configurations the hook observes the
    exact head input chosen by that model. Padding never becomes an event.
    """
    from transformer_ee.dataloader.pd_dataset import Normalized_pandas_Dataset_with_cache
    required=config['vector']+config['scalar']
    missing=set(required)-set(frame.columns)
    if missing:raise ValueError(f'Missing model inputs: {sorted(missing)}')
    # Validate BEFORE the upstream string parser could replace absent data by zero.
    for col in config['scalar']:
        if not np.isfinite(pd.to_numeric(frame[col]).to_numpy(float)).all():
            raise ValueError(f'Nonfinite scalar {col}')
    lengths=[]
    for _,row in frame.iterrows():
        counts=[]
        for col in config['vector']:
            if not isinstance(row[col],str) or not row[col]:raise ValueError(f'Empty vector {col}')
            vector=np.array([float(x) for x in row[col].split(',')])
            if not np.isfinite(vector).all():raise ValueError(f'Nonfinite vector {col}')
            counts.append(len(vector))
        if len(set(counts))!=1:raise ValueError('Particle vector lengths differ within an event')
        lengths.append(counts[0])
    for col in required:
        if col not in stats or len(stats[col])!=2 or not np.isfinite(stats[col]).all() or stats[col][1]<=0:
            raise ValueError(f'Invalid saved normalization for {col}')
    dataset=Normalized_pandas_Dataset_with_cache(config,frame.copy(deep=True),eval=True,use_cache=False)
    dataset.normalize(stats)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    head,penultimate,predictions=[],[],[]
    head_module=net.new_head if net.use_new_head else net.linear1
    final_module=net.new_head[-1] if net.use_new_head else net.linear3
    def head_hook(module,args):head.append(args[0].detach().cpu().numpy().copy())
    def final_hook(module,args):penultimate.append(args[0].detach().cpu().numpy().copy())
    handles=[head_module.register_forward_pre_hook(head_hook),final_module.register_forward_pre_hook(final_hook)]
    try:
        with torch.inference_mode():
            for vectors,scalars,mask,_,_ in loader:
                out=net(vectors.to(device),scalars.to(device),mask.to(device))
                predictions.append(out.cpu().numpy())
    finally:
        for handle in handles:handle.remove()
    arrays=[np.concatenate(x) for x in (head,penultimate,predictions)]
    if any(len(x)!=len(frame) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Invalid output or activation/event alignment')
    return *arrays,dict(truncated_events=int(np.sum(np.array(lengths)>config['max_num_prongs'])),
                       max_observed_prongs=max(lengths),max_num_prongs=config['max_num_prongs'])



ORIGINAL_FORWARD = forward_frame

def prepare_polars(config, stats, frame):
    """Vectorize parsing, normalization, truncation, and zero padding in Polars.

    Keep Float64 until normalization is finished, then cast the network inputs
    to Float32. The older repository Polars Dataset casts earlier; importing it
    unchanged would introduce a different rounding order. Metadata/selection
    remains in the shared pandas cohort, and no Arrow dependency is required.
    """
    import polars as pl
    vectors, scalars = config['vector'], config['scalar']
    width = config['max_num_prongs']
    for name in vectors+scalars:
        mean, std = stats[name]
        if not np.isfinite([mean,std]).all() or std <= 0:
            raise ValueError('Invalid normalization')
    data = pl.DataFrame([pl.Series(v,frame[v].tolist(),dtype=pl.String) for v in vectors] +
                        [pl.Series(s,frame[s].to_numpy(float)) for s in scalars])
    data = data.with_columns([pl.col(v).str.split(',').list.eval(
        (pl.element().cast(pl.Float64)-stats[v][0])/stats[v][1]) for v in vectors])
    lengths = data[vectors[0]].list.len().to_numpy()
    if any(not np.array_equal(lengths,data[v].list.len().to_numpy()) for v in vectors):
        raise ValueError('Misaligned input vectors')
    # Append normalized-space zeros only AFTER normalizing physical particles.
    # Truncate to the checkpoint's width and preserve original prong counts.
    x = np.stack([data.select(pl.col(v).list.head(width).list.concat(pl.lit([0.0]*width))
        .list.head(width).list.to_array(width).cast(pl.Array(pl.Float32,width)))
        .to_series().to_numpy() for v in vectors],axis=2)
    y = data.select([((pl.col(s)-stats[s][0])/stats[s][1]).cast(pl.Float32).alias(s)
                     for s in scalars]).to_numpy()
    return x, y, np.arange(width)[None,:]>=lengths[:,None], lengths


def forward_fast(net,config,stats,frame,device='cpu',batch_size=256,prepare_engine='pandas'):
    """Batch the exact saved normalization and original model forward pass.

    The old Dataset reparses pandas rows for each event. Here we prepare the
    same padded float32 arrays once. All padding, truncation and mask semantics
    match that Dataset. Every model/source call is checked against the original
    implementation on 32 records, including head and penultimate activations.
    """
    n=len(frame); width=config['max_num_prongs']; vectors=config['vector']; scalars=config['scalar']
    if prepare_engine=='polars':
        x,y,mask,lengths=prepare_polars(config,stats,frame)
    elif prepare_engine=='pandas':
        x=np.zeros((n,width,len(vectors)),np.float32)
        lengths=None
        for j,col in enumerate(vectors):
            values=[np.asarray([float(v) for v in text.split(',')]) for text in frame[col]]
            sizes=np.asarray([len(v) for v in values])
            if lengths is None: lengths=sizes
            elif not np.array_equal(lengths,sizes): raise ValueError('Misaligned input vectors')
            mean,std=stats[col]
            if not np.isfinite([mean,std]).all() or std<=0: raise ValueError('Invalid normalization')
            for i,value in enumerate(values): x[i,:min(width,len(value)),j]=((value-mean)/std)[:width]
        y=np.column_stack([(frame[c].to_numpy(float)-stats[c][0])/stats[c][1] for c in scalars]).astype(np.float32)
        mask=np.arange(width)[None,:]>=lengths[:,None]
    else:
        raise ValueError('Unknown preparation engine')
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite normalized inputs')
    heads=[]; pens=[]; preds=[]
    first=net.new_head if net.use_new_head else net.linear1
    last=net.new_head[-1] if net.use_new_head else net.linear3
    handles=[first.register_forward_pre_hook(lambda module,args:heads.append(args[0].detach().cpu().numpy().copy())),
             last.register_forward_pre_hook(lambda module,args:pens.append(args[0].detach().cpu().numpy().copy()))]
    # Process similar-length events together and omit columns that are padding
    # for the entire batch. The original attention mask excludes those columns
    # and pooling zeroes them, so no physical prong is removed. Restore original
    # file order before exporting. This is substantial on mostly 3-prong events.
    order=np.argsort(lengths,kind='stable')
    start=time.monotonic(); last_report=start
    try:
        with torch.inference_mode():
            for a in range(0,n,batch_size):
                b=min(a+batch_size,n)
                ids=order[a:b]; used=max(1,min(width,int(lengths[ids].max())))
                pred=net(torch.from_numpy(x[ids,:used]).to(device),torch.from_numpy(y[ids]).to(device),
                         torch.from_numpy(mask[ids,:used]).to(device))
                preds.append(pred.cpu().numpy())
                if time.monotonic()-last_report>45:
                    print(f'  forward {b:,}/{n:,} events ({time.monotonic()-start:.0f}s)',flush=True)
                    last_report=time.monotonic()
    finally:
        for handle in handles:handle.remove()
    inverse=np.argsort(order)
    arrays=[np.concatenate(v)[inverse] for v in (heads,pens,preds)]
    if any(not np.isfinite(v).all() for v in arrays): raise ValueError('Nonfinite model outputs')
    # Include every observed prong length as well as the leading 32 events; rare
    # long sequences and truncation boundaries must also match the legacy path.
    check_ids=np.unique(np.r_[np.arange(min(32,n)),[np.flatnonzero(lengths==v)[0] for v in np.unique(lengths)]])
    reference=ORIGINAL_FORWARD(net,config,stats,frame.iloc[check_ids].reset_index(drop=True),device,64)
    errors=[]
    for actual,expected in zip(arrays,reference[:3]):
        np.testing.assert_allclose(actual[check_ids],expected,rtol=2e-4,atol=1e-5)
        errors.append(float(np.max(np.abs(actual[check_ids]-expected))))
    return *arrays,dict(truncated_events=int((lengths>width).sum()),max_observed_prongs=int(lengths.max()),
        max_num_prongs=width,original_forward_check_n=len(check_ids),original_forward_max_abs_errors=errors,
        batching='stable prong-count order; all-padding columns omitted; original event order restored')
