"""
A callable class to add noise to the input data of model during training.
Please read the following note before using this class.

Why callable?
- Because collate_fn in DataLoader should be callable.

What exactly does this class do?
- This class performs a LINEAR TRANSFORMATION on the input data, which is equivalent to:
    1. Multiplying the un-normalized input data by a random number centered at 1.
    2. Normalizing it. (transformer_EE always normalizes the input data)

In transformer_EE, the input data is normalized and cached to avoid repeating the normalization 
process. This makes adding noise to the input data a bit tricky if we want the noise to be different 
for each epoch. Thus, we need to trace back the normalization process and add noise to the 
normalized input data.

Assuming we intend to perform the following transformation on the un-normalized input:
    x -> x * (1 + noise)
and x is normalized to x' by the following formula:
    x' = (x - mean) / std
where mean and std are the mean and standard deviation of the training set, respectively.
Reverse the normalization process to get x:
    x = x' * std + mean
then we have:
    x' -> (x * (1 + noise) - mean) / std
        = ((x' * std + mean) * (1 + noise) - mean) / std
        = x' * (1 + noise) + noise * mean / std
"""

import csv
import os
import torch
from torch.utils.data._utils.collate import default_collate


class normalized_noise:
    """
    Use this with Normalized_pandas_Dataset_with_cache.
    """

    def __init__(self, config, stats):
        """
        config:     the configuration dictionary
        stats:      the statistics of the training set; a dictionary with keys
                    "vector" and "scalar" containing the mean and std of the training
        example of config:
        {
            "vector": ["sequence1", "sequence2"],
            "scalar": ["scalar1", "scalar2"],
            "noise": {
                "name": "gaussian",
                "mean": 0,
                "std": 0.1,
                "vector": ["sequence1"],
                "scalar": ["scalar1"],
            }
        }
        """
        if "noise" not in config:
            raise ValueError("Noise configuration not found!")
        self.noise = None
        if config["noise"]["name"] == "gaussian":
            self.noise = lambda size: torch.normal(
                mean=config["noise"]["mean"], std=config["noise"]["std"], size=size
            )
        elif config["noise"]["name"] == "uniform":
            self.noise = lambda size: config["noise"]["low"] + torch.rand(size=size) * (
                config["noise"]["high"] - config["noise"]["low"]
            )
        else:
            raise NotImplementedError("Noise distribution not implemented!")

        # get indices of vector, scalar that need to be added noise
        self.vector_indices = [
            config["vector"].index(name) for name in config["noise"]["vector"]
        ]
        self.scalar_indices = [
            config["scalar"].index(name) for name in config["noise"]["scalar"]
        ]
        # find the indices of the vector and scalar names
        # calculate mean / std for the vector and scalar
        self.vector_name_indices = []
        self.scalar_name_indices = []
        self.vector_correction = []
        self.scalar_correction = []
        for name in config["noise"].get("vector", []):
            self.vector_name_indices.append(config["vector"].index(name))
            self.vector_correction.append(stats[name][0] / stats[name][1])
        for name in config["noise"].get("scalar", []):
            self.scalar_name_indices.append(config["scalar"].index(name))
            self.scalar_correction.append(stats[name][0] / stats[name][1])
        self.vector_correction = torch.Tensor(self.vector_correction)
        self.scalar_correction = torch.Tensor(self.scalar_correction)
        self.vector_names = list(config["vector"])
        self.scalar_names = list(config["scalar"])
        self.vector_mean = torch.tensor(
            [stats[name][0] for name in self.vector_names], dtype=torch.float32
        )
        self.vector_std = torch.tensor(
            [stats[name][1] for name in self.vector_names], dtype=torch.float32
        )
        self.scalar_mean = torch.tensor(
            [stats[name][0] for name in self.scalar_names], dtype=torch.float32
        )
        self.scalar_std = torch.tensor(
            [stats[name][1] for name in self.scalar_names], dtype=torch.float32
        )

        export_cfg = config.get("noise_export", {})
        self.export_enabled = bool(export_cfg.get("enabled", False))
        self.export_epochs = int(export_cfg.get("epochs", 1))
        self.export_path = export_cfg.get("path")
        self.export_include_normalized = bool(
            export_cfg.get("include_normalized", False)
        )
        self.current_epoch = 0
        self.batch_in_epoch = 0
        self._header_written = False
        self._export_announced = False

    def configure_output_dir(self, output_dir):
        """
        Provide a default export path once the trainer has resolved save_path.
        """
        if self.export_enabled and not self.export_path:
            self.export_path = os.path.join(output_dir, "noised_training_inputs.csv")

    def set_epoch(self, epoch):
        """
        Reset epoch-local counters so exported rows carry stable metadata.
        """
        self.current_epoch = epoch
        self.batch_in_epoch = 0

    def _should_export_current_epoch(self):
        return self.export_enabled and self.current_epoch < self.export_epochs

    def _ensure_export_ready(self):
        if not self._should_export_current_epoch():
            return False
        if not self.export_path:
            raise ValueError(
                "noise_export is enabled but no export path is configured."
            )
        os.makedirs(os.path.dirname(os.path.abspath(self.export_path)), exist_ok=True)
        if not self._export_announced:
            print(f"[INFO] Exporting noised training inputs to {self.export_path}")
            self._export_announced = True
        return True

    def _denormalize(self, vector, scalar):
        raw_vector = (
            vector * self.vector_std[None, None, :] + self.vector_mean[None, None, :]
        )
        raw_scalar = scalar * self.scalar_std[None, :] + self.scalar_mean[None, :]
        return raw_vector, raw_scalar

    def _format_sequence(self, values):
        return ",".join(f"{float(v):.10g}" for v in values)

    def _write_export_rows(self, vector, scalar, mask, sample_indices, noise_values):
        if not self._ensure_export_ready():
            return

        raw_vector, raw_scalar = self._denormalize(vector, scalar)
        header = ["epoch", "batch_in_epoch", "dataset_index", "noise_draw", "seq_len"]
        header.extend(self.scalar_names)
        header.extend(self.vector_names)
        if self.export_include_normalized:
            header.extend(f"{name}__normalized" for name in self.scalar_names)
            header.extend(f"{name}__normalized" for name in self.vector_names)

        mode = "a" if self._header_written else "w"
        with open(self.export_path, mode, newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not self._header_written:
                writer.writerow(header)
                self._header_written = True

            mask_cpu = mask.detach().cpu()
            raw_vector_cpu = raw_vector.detach().cpu()
            raw_scalar_cpu = raw_scalar.detach().cpu()
            vector_cpu = vector.detach().cpu()
            scalar_cpu = scalar.detach().cpu()
            sample_indices_cpu = sample_indices.detach().cpu()
            noise_cpu = noise_values.detach().cpu()

            for row_idx in range(raw_vector_cpu.shape[0]):
                seq_len = int((~mask_cpu[row_idx]).sum().item())
                row = [
                    self.current_epoch,
                    self.batch_in_epoch,
                    int(sample_indices_cpu[row_idx].item()),
                    float(noise_cpu[row_idx].item()),
                    seq_len,
                ]
                row.extend(float(x) for x in raw_scalar_cpu[row_idx].tolist())
                for feat_idx in range(raw_vector_cpu.shape[2]):
                    row.append(
                        self._format_sequence(
                            raw_vector_cpu[row_idx, :seq_len, feat_idx].tolist()
                        )
                    )
                if self.export_include_normalized:
                    row.extend(float(x) for x in scalar_cpu[row_idx].tolist())
                    for feat_idx in range(vector_cpu.shape[2]):
                        row.append(
                            self._format_sequence(
                                vector_cpu[row_idx, :seq_len, feat_idx].tolist()
                            )
                        )
                writer.writerow(row)

    def __call__(self, batch):
        """
        batch: the normalized input data
        return: a tuple of noisy input data and target data
        """

        # default_collate to convert the list of tensors to a tensor
        batch = default_collate(batch)
        sample_indices = None
        if len(batch) == 6:
            vector, scalar, mask, target, weight, sample_indices = batch
        else:
            vector, scalar, mask, target, weight = batch
        # shape of vector: (batch_size, max_seq_len, vector_dim)
        # shape of scalar: (batch_size, scalar_dim)
        # shape of mask: (batch_size, max_seq_len)
        # shape of target: (batch_size, target_dim)
        # shape of weight: (batch_size, 1)
        _noise = self.noise(size=(vector.shape[0],))
        # apply noise to the vector
        if self.vector_indices:
            vector[:, :, self.vector_indices] += (
                vector[:, :, self.vector_indices] * _noise[:, None, None]
                + self.vector_correction[None, None, :] * _noise[:, None, None]
            )
        # apply noise to the scalar
        if self.scalar_indices:
            scalar[:, self.scalar_indices] += (
                scalar[:, self.scalar_indices] * _noise[:, None]
                + self.scalar_correction[None, :] * _noise[:, None]
            )
        if sample_indices is not None:
            self._write_export_rows(vector, scalar, mask, sample_indices, _noise)
        self.batch_in_epoch += 1
        # NOTE: after adding noise, the padding values may not be 0 anymore,
        # but it should not matter
        return vector, scalar, mask, target, weight
