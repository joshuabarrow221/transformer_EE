#!/usr/bin/env python3
"""Compare CSV loading and vector normalization in independent worker processes.

This is a preparation microbenchmark, not an alternative scientific pipeline.
It does not time complete topology audits, padding, GPU inference, or t-SNE.
Use the same source prefix, saved normalization, and thread count for each engine.
No source or checkpoint files are modified. Polars is an optional dependency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    config = json.loads((args.model_directory/'input.json').read_text())
    stats = json.loads((args.model_directory/'trainset_stat.json').read_text())
    vectors, scalars = config['vector'], config['scalar']
    columns = list(dict.fromkeys(vectors+scalars+['Topology','Event_Index']))
    if args.engine == 'pandas':
        import pandas as pd
        version = pd.__version__
        start = time.perf_counter()
        frame = pd.read_csv(args.csv,nrows=args.rows,usecols=columns,
                            dtype={'Topology':str},float_precision='round_trip')
        read_seconds = time.perf_counter()-start
        start = time.perf_counter()
        arrays = {}
        for name in vectors:
            mean, std = stats[name]
            arrays[name] = [(np.asarray([float(x) for x in text.split(',')])-mean)/std
                            if isinstance(text,str) else np.array([np.nan]) for text in frame[name]]
        scalar_array = np.column_stack([(frame[s].to_numpy(float)-stats[s][0])/stats[s][1] for s in scalars])
        prepare_seconds = time.perf_counter()-start
        ids = frame.Event_Index.to_numpy(np.int64)
        topology = frame.Topology.tolist()
        flattened = lambda name: np.concatenate(arrays[name])
        lengths = lambda name: np.array([len(a) for a in arrays[name]],dtype=np.int64)
    else:
        import polars as pl
        version = pl.__version__
        start = time.perf_counter()
        frame = pl.read_csv(args.csv,n_rows=args.rows,columns=columns,
                            schema_overrides={'Topology':pl.String},n_threads=args.threads).head(args.rows)
        read_seconds = time.perf_counter()-start
        start = time.perf_counter()
        # Float64 parsing/normalization matches the pandas reference's order.
        # The existing training Polars Dataset casts to Float32 earlier instead.
        frame = frame.with_columns([
            pl.col(v).str.split(',').list.eval(
                (pl.element().cast(pl.Float64,strict=False)-stats[v][0])/stats[v][1])
            for v in vectors] + [
            ((pl.col(s).cast(pl.Float64)-stats[s][0])/stats[s][1]).alias(s) for s in scalars])
        scalar_array = frame.select(scalars).to_numpy()
        prepare_seconds = time.perf_counter()-start
        ids = frame['Event_Index'].to_numpy().astype(np.int64)
        topology = frame['Topology'].to_list()
        flattened = lambda name: frame[name].explode(empty_as_null=True).to_numpy()
        lengths = lambda name: frame[name].list.len().fill_null(1).to_numpy().astype(np.int64)
    # Verify actual prepared values at the network's Float32 input precision,
    # not merely row counts. Float64 expression evaluation may differ by an ULP
    # between NumPy and Polars even when network inputs are identical. Hashing
    # is outside the timed phases but included in the RSS high-water mark.
    hashes = {}
    for name in vectors:
        values = np.asarray(flattened(name),dtype='<f4').copy()
        values[np.isnan(values)] = np.nan
        h = hashlib.sha256(lengths(name).tobytes())
        h.update(values.tobytes())
        hashes[name] = h.hexdigest()
    scalar_array = np.asarray(scalar_array,dtype='<f4').copy()
    scalar_array[np.isnan(scalar_array)] = np.nan
    hashes['scalars'] = hashlib.sha256(scalar_array.tobytes()).hexdigest()
    hashes['event_ids'] = hashlib.sha256(ids.tobytes()).hexdigest()
    hashes['topology'] = hashlib.sha256('\n'.join(topology).encode()).hexdigest()
    return dict(engine=args.engine,version=version,rows=len(frame),threads=args.threads,
                read_seconds=read_seconds,normalize_seconds=prepare_seconds,
                total_seconds=read_seconds+prepare_seconds,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                prepared_hashes=hashes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--model-directory',type=Path,required=True)
    p.add_argument('--rows',type=int,default=199990)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path)
    p.add_argument('--engine',choices=['both','pandas','polars'],default='both',help='Individual engines are internal workers')
    args = p.parse_args()
    if min(args.rows,args.threads,args.repeats) < 1:
        p.error('Rows, threads, and repeats must be positive')
    if args.engine != 'both':
        print(json.dumps(worker(args)))
        return
    results = []
    environment = dict(os.environ,POLARS_MAX_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads))
    for repeat in range(args.repeats):
        # Alternate order to reduce systematic cache/order advantages. Do not
        # drop OS caches: timings represent ordinary repeated local reads.
        for engine in (['pandas','polars'] if repeat%2==0 else ['polars','pandas']):
            command = [sys.executable,str(Path(__file__).resolve()),'--engine',engine,
                       '--csv',str(args.csv),'--model-directory',str(args.model_directory),
                       '--rows',str(args.rows),'--threads',str(args.threads)]
            result = json.loads(subprocess.check_output(command,env=environment,text=True))
            results.append(result)
            print(engine,repeat+1,f'{result["total_seconds"]:.3f}s',f'{result["peak_rss_mib"]:.1f} MiB peak RSS',flush=True)
    exact = all(r['prepared_hashes']==results[0]['prepared_hashes'] and r['rows']==results[0]['rows'] for r in results)
    summary = {engine:{key:statistics.median(r[key] for r in results if r['engine']==engine)
               for key in ('read_seconds','normalize_seconds','total_seconds','peak_rss_mib')}
               for engine in ('pandas','polars')}
    report = dict(scope='CSV prefix reading and vector/scalar normalization only',
                  exact_prepared_float32_values_and_ids=exact,
                  comparison_dtype='float32, matching network input precision',summary=summary,runs=results)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
    if not exact:
        raise SystemExit('Prepared outputs differ: do not interpret timing as an equivalent pipeline comparison')


if __name__ == '__main__':
    main()
