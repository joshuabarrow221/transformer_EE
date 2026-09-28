# Manifests and private asset maps

`quickstart.json` selects one verified MV checkpoint and beam AR23 source.
`study.json` records the current full study, including the later completed SV
noise configurations. Their relative asset paths can be changed without editing
Python. Neither file contains a user's absolute filesystem paths.

The manifest schema is deliberately small:

```json
{
  "version": 1,
  "models": {
    "my_network": {"directory": "my_checkpoint_folder"}
  },
  "sources": {
    "my_events": {"path": "my_events.csv", "generator": "AR23"}
  },
  "groups": [{
    "id": "my_grid",
    "title": "My MV comparison",
    "training_generator": "AR23",
    "inference_label": "Natural-spectrum inference",
    "rows": {
      "Flat_NoNoise": {"models": ["my_network"], "sources": ["my_events"]}
    }
  }]
}
```

Model `directory` is relative to `--models-root`; source `path` is relative to
`--samples-root`. A model directory must contain `best_model.zip`, `input.json`,
and `trainset_stat.json`. Optional `sha256` maps those three filenames to expected
hashes. The shipped study pins all three; custom manifests may omit expected
hashes, but the runner still records actual hashes and checks them on resume.

Rows use `Flat_Noise`, `Flat_NoNoise`, `Natural_Noise`, `Natural_NoNoise`.
The established plotting layout supports generator columns `AR23`, `G2111a`,
`G1810a0211a`, `G1810a0211b`, `NuWro`; use the canonical names, not filename aliases.
There is one source per generator per row. A row has one MV checkpoint or four
SV components, with every physical target covered exactly once. List SV networks
in E/Px/Py/Pz order for readable provenance. All sources in a row use that same
bundle. Missing cells can be explained with an optional `missing` mapping whose
keys have the form `Natural_NoNoise/NuWro`.

`training_generator` and `inference_label` are explicit annotations supplied by
the manifest author; filenames do not establish that provenance. The exporter
does not infer or choose new checkpoint replicas. Group IDs are safe filenames;
absolute paths and paths escaping an asset root are rejected.

## Private transfer inventory

`stage_assets.py` takes a separate inventory, kept outside Git. Its entries map
your local source tree to the manifest's desired relative layout:

```json
{
  "version": 1,
  "assets": [
    {"kind": "sample", "source": "local_data/events.csv",
     "destination": "my_events.csv", "bytes": 12345},
    {"kind": "model", "source": "local_runs/run1/best_model.zip",
     "destination": "my_checkpoint_folder/best_model.zip", "bytes": 54321}
  ]
}
```

Add corresponding entries for `input.json` and `trainset_stat.json`, and supply
actual byte counts. `sha256` is optional in the inventory; checkpoint entries
should pin it. Sources are resolved below `--source-root`. Destinations are below
`--destination/samples` or `--destination/models`. The collector only copies assets
referenced by the chosen manifest. Its default is a dry run; `--copy` performs
verified copies and writes `TRANSFER.json`. It never deletes or overwrites a
different existing asset. Transfer that folder to the execution host separately.
