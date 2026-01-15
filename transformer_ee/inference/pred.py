"""
This module contains the functions to make predictions using the trained model, AKA inference.
It shares the same data loading and preprocessing functions with the training module. 
However, the target values are not provided in the input data and weights are not calculated. See transformer_ee/dataloader/pd_dataset.py for more details.
"""

import os
import json
import pandas as pd
import torch
import numpy as np
import time
from transformer_ee.utils import get_gpu
from transformer_ee.dataloader.pd_dataset import Normalized_pandas_Dataset_with_cache
from .load_model_checkpoint import load_model_checkpoint


def _resolve_device(device):
    """
    Helper to choose the correct torch.device
    """

    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return get_gpu()


class Predictor:
    """
    A class to make predictions using a trained model.
    """

    def __init__(self, model_dir: str, dtframe: pd.DataFrame, device=None, num_workers=None):
        self.device = _resolve_device(device)
        self.model_dir = model_dir
        self.original_df = dtframe.copy()  # keep untouched snapshot for reporting
        with open(
            os.path.join(self.model_dir, "input.json"), encoding="UTF-8", mode="r"
        ) as fc:
            self.train_config = json.load(fc)
        print("Loading model...")
        self.net = load_model_checkpoint(self.model_dir)
        self.net.eval()
        self.net.to(self.device)
        print("Preparing dataframe...")
        self.df = dtframe.copy()
        self._ensure_required_columns(self.df)
        print("Loading dataset...")
        self.dataset = Normalized_pandas_Dataset_with_cache(
            self.train_config,
            self.df,
            eval=True,
            use_cache=False,
        )
        self.train_stat = {}
        with open(
            os.path.join(self.model_dir, "trainset_stat.json"),
            encoding="UTF-8",
            mode="r",
        ) as fs:
            self.train_stat = json.load(fs)
        self.dataset.normalize(self.train_stat)
        print("Creating dataloader...")
        worker_count = (
            num_workers
            if num_workers is not None
            else self.train_config.get("num_workers", 10)
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.train_config["batch_size_test"],
            shuffle=False,
            num_workers=worker_count,
        )

    def _ensure_required_columns(self, df: pd.DataFrame):
        vectors = self.train_config["vector"]
        scalars = self.train_config["scalar"]
        missing_vectors = [c for c in vectors if c not in df.columns]
        missing_scalars = [c for c in scalars if c not in df.columns]
        for col in missing_vectors:
            df[col] = None  # string_to_float_list will convert None -> [0]
        for col in missing_scalars:
            df[col] = 0.0
        if missing_vectors or missing_scalars:
            print(
                f"[WARN] Added placeholder values for missing columns. "
                f"Vectors: {missing_vectors or 'none'}, Scalars: {missing_scalars or 'none'}"
            )

    def go(self, return_batch_times: bool = False):
        """
        A function to make predictions using the trained model.
        """
        prediction = []
        batch_times = []
        batch_sizes = []
        total = len(self.dataset)
        processed = 0
        with torch.no_grad():
            for _batch_idx, batch in enumerate(self.dataloader):
                batch_start = time.perf_counter()
                processed = min(total, processed + batch[0].shape[0])
                batch_sizes.append(batch[0].shape[0])
                percent = 100.0 * processed / total if total else 0.0
                print(
                    f"Batch { _batch_idx + 1 } / { len(self.dataloader) }"
                    f" | processed {processed}/{total} ({percent:.2f}% complete)"
                )
                vector_valid_batch = batch[0].to(self.device)
                scalar_valid_batch = batch[1].to(self.device)
                mask_valid_batch = batch[2].to(self.device)
                Netout = self.net.forward(
                    vector_valid_batch, scalar_valid_batch, mask_valid_batch
                )
                prediction.append((Netout.detach().cpu().numpy()))
                batch_times.append(time.perf_counter() - batch_start)
        prediction = np.concatenate(prediction)
        if return_batch_times:
            return prediction, batch_times, batch_sizes
        return prediction
