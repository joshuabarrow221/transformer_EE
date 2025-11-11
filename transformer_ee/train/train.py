# ========================= train.py =========================

"""A module for training a model – revised with richer bookkeeping."""

import json
import os
import math
from typing import Dict, List

import numpy as np
import torch
import pandas as pd

from transformer_ee.dataloader.load import get_train_valid_test_dataloader
from transformer_ee.logger.train_logger import BaseLogger
from transformer_ee.model import create_model
from transformer_ee.utils import get_gpu, hash_dict

from .loss import (
    linear_combination_loss,
    abs_invariant_mass_squared_prediction_loss,
    loss_function,  # expose base losses for per‑component tracking
)
from .loss_track import plot_loss
from .optimizer import create_optimizer

# -----------------------------------------------------------------------------
# Alpha / Beta schedules (unchanged)
# -----------------------------------------------------------------------------

def alpha(epoch: float, a_start: float = 1.0, a_end: float = 1.0, k: float = 1.0, x0: float = 6.0) -> float:
    """Decreasing S‑curve from a_start→a_end over ~10 epochs by default."""
    return a_end + (a_start - a_end) / (1 + math.exp(k * (epoch - x0)))


def beta(epoch: float, b_start: float = 0.01, b_end: float = 1.0, k: float = 1.0, x0: float = 6.0) -> float:
    """Increasing S‑curve from b_start→b_end over ~10 epochs by default."""
    return b_start + (b_end - b_start) / (1 + math.exp(-k * (epoch - x0)))

# -----------------------------------------------------------------------------
# Trainer class
# -----------------------------------------------------------------------------

class MVtrainer:
    r"""
    Multivariate trainer with detailed loss breakdown.
    """

    # ------------------------------------------------------------------
    # construction & bookkeeping containers
    # ------------------------------------------------------------------

    def __init__(self, input_d: dict, logger: BaseLogger | None = None):
        self.gpu_device = get_gpu()
        self.input_d = input_d
        self.logger = logger
        print(json.dumps(self.input_d, indent=4))
        # A single, ever-increasing step counter for W&B
        self.global_step = 0

        # dataloaders & model
        (
            self.trainloader,
            self.validloader,
            self.testloader,
            self.train_set_stat,
        ) = get_train_valid_test_dataloader(input_d)
        self.net = create_model(self.input_d).to(self.gpu_device)
        self.optimizer = create_optimizer(self.input_d, self.net)
        self.bestscore = 1e9
        self.print_interval = self.input_d.get("print_interval", 1)

        # epoch‑wise aggregates (legacy)
        self.train_loss_list_per_epoch: List[float] = []
        self.valid_loss_list_per_epoch: List[float] = []

        # NEW – component‑wise LCL contributions (shape = [epochs, n_vars])
        self.train_comp_loss_per_epoch: List[torch.Tensor] = []
        self.valid_comp_loss_per_epoch: List[torch.Tensor] = []

        # NEW – specialised trackers for PredInvMass scheme
        self.alpha_list: List[float] = []
        self.beta_list: List[float] = []
        self.train_pred_mass_per_epoch: List[float] = []
        self.valid_pred_mass_per_epoch: List[float] = []
        self.train_alpha_LCL_per_epoch: List[float] = []
        self.train_beta_PM_per_epoch: List[float] = []
        self.valid_alpha_LCL_per_epoch: List[float] = []
        self.valid_beta_PM_per_epoch: List[float] = []

        # misc
        self.epochs = input_d["model"].get("epochs", 100)
        print("epochs: ", self.epochs)
        self.save_path = os.path.join(input_d.get("save_path", "."), "model_" + hash_dict(self.input_d))
        os.makedirs(self.save_path, exist_ok=True)
        self._dump_static_json()

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------

    def _dump_static_json(self):
        input_json = os.path.join(self.save_path, "input.json")
        if not os.path.exists(input_json):
            with open(input_json, "w") as f:
                json.dump(self.input_d, f, indent=4)

        stat_json = os.path.join(self.save_path, "trainset_stat.json")
        if not os.path.exists(stat_json):
            with open(stat_json, "w") as f:
                json.dump(self.train_set_stat, f, indent=4)

    # ------------------------------------------------------------------
    # core training loops – LCL only
    # ------------------------------------------------------------------

    def train_LCL(self):
        """Standard LCL training with per‑component loss tracking."""
        loss_kwargs = self.input_d["loss"]["kwargs"]
        base_loss_names = loss_kwargs["base_loss_names"]
        coeffs = loss_kwargs["coefficients"]
        n_vars = len(base_loss_names)

        for epoch in range(self.epochs):
            # ------------------- training -------------------
            self.net.train()
            epoch_total_loss = []
            comp_sum = torch.zeros(n_vars, device=self.gpu_device)

            for batch_idx, batch in enumerate(self.trainloader):
                vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]

                out = self.net.forward(vec, sca, msk)

                # total loss (legacy)
                lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)

                # component‑wise contributions (un‑weighted by alpha/beta)
                for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                    comp_sum[j] += (
                        coef
                        * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                        .detach()
                    )

                # step
                self.optimizer.zero_grad()
                lcl_loss.backward()
                self.optimizer.step()

                epoch_total_loss.append(lcl_loss.detach())
                # ---- W&B per-step logging (main process only) ----
                if self.logger is not None:
                    # ensure plain Python floats; also log a monotonic step
                    self.logger.log({
                        "train/loss": float(lcl_loss.detach().item()),
                        "train/step": int(self.global_step),
                        "epoch": int(epoch),
                    }, step=int(self.global_step))
                self.global_step += 1


                if batch_idx % self.print_interval == 0:
                    print(f"Epoch: {epoch}, batch: {batch_idx} Loss: {lcl_loss:0.4f}")

            # epoch aggregates
            self.train_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_total_loss)))
            self.train_comp_loss_per_epoch.append(comp_sum / len(self.trainloader))

            # ------------------- validation -------------------
            self.net.eval()
            epoch_val_loss = []
            comp_val_sum = torch.zeros(n_vars, device=self.gpu_device)

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validloader):
                    vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                    out = self.net.forward(vec, sca, msk)

                    lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)

                    for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                        comp_val_sum[j] += (
                            coef
                            * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                            .detach()
                        )

                    epoch_val_loss.append(lcl_loss.detach())

            self.valid_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_val_loss)))
            self.valid_comp_loss_per_epoch.append(comp_val_sum / len(self.validloader))

            # ---- Optional: epoch-level summaries (once per epoch) ----
            if self.logger is not None:
                tr = float(self.train_loss_list_per_epoch[-1].detach().cpu().item()
                           if hasattr(self.train_loss_list_per_epoch[-1], "detach") else
                           float(self.train_loss_list_per_epoch[-1]))
                va = float(self.valid_loss_list_per_epoch[-1].detach().cpu().item()
                           if hasattr(self.valid_loss_list_per_epoch[-1], "detach") else
                           float(self.valid_loss_list_per_epoch[-1]))
                # batch components (example: log as a small dict)
                comp = self.train_comp_loss_per_epoch[-1].detach().cpu().tolist()
                comp_dict = {f"train/comp_{j}": float(x) for j, x in enumerate(comp)}
                self.logger.log({
                    "epoch": int(epoch),
                    "train/loss_epoch": tr,
                    "valid/loss_epoch": va,
                    **comp_dict,
                    "train/step": int(self.global_step),  # keeps charts aligned
                }, step=int(self.global_step))

            # ------------------- bookkeeping & plots -------------------
            self._checkpoint_and_plot(epoch, scheme="LCL", n_vars=n_vars)

        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------
    # core training loops – LCL + PredInvMass
    # ------------------------------------------------------------------

    def train_LCL_wPredInvMass(self):
        """Hybrid loss training with richer analytics."""
        loss_kwargs = self.input_d["loss"]["kwargs"]
        base_loss_names = loss_kwargs["base_loss_names"]
        coeffs = loss_kwargs["coefficients"]
        n_vars = len(base_loss_names)

        for epoch in range(self.epochs):
            a_val = alpha(epoch)
            b_val = beta(epoch)
            self.alpha_list.append(a_val)
            self.beta_list.append(b_val)

            # ------------------- training -------------------
            self.net.train()
            epoch_total_loss = []
            comp_sum = torch.zeros(n_vars, device=self.gpu_device)
            pm_sum = 0.0
            aLCL_sum = 0.0
            bPM_sum = 0.0

            for batch_idx, batch in enumerate(self.trainloader):
                vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                out = self.net.forward(vec, sca, msk)

                lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                pm_loss = abs_invariant_mass_squared_prediction_loss(out, weight=wgt)
                total_loss = a_val * lcl_loss + b_val * pm_loss

                # per component contribution of LCL (for diagnostics)
                for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                    comp_sum[j] += (
                        coef
                        * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                        .detach()
                    )

                # step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss.append(total_loss.detach())
                pm_sum += pm_loss.detach()
                aLCL_sum += (a_val * lcl_loss).detach()
                bPM_sum += (b_val * pm_loss).detach()

                if batch_idx % self.print_interval == 0:
                    print(f"Epoch: {epoch}, batch: {batch_idx} Loss: {total_loss:0.4f}")

            # epoch aggregates (train)
            n_train_batches = len(self.trainloader)
            self.train_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_total_loss)))
            self.train_comp_loss_per_epoch.append(comp_sum / n_train_batches)
            self.train_pred_mass_per_epoch.append(pm_sum / n_train_batches)
            self.train_alpha_LCL_per_epoch.append(aLCL_sum / n_train_batches)
            self.train_beta_PM_per_epoch.append(bPM_sum / n_train_batches)

            # ------------------- validation -------------------
            self.net.eval()
            epoch_val_loss = []
            comp_val_sum = torch.zeros(n_vars, device=self.gpu_device)
            pm_val_sum = 0.0
            aLCL_val_sum = 0.0
            bPM_val_sum = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validloader):
                    vec, sca, msk, tgt, wgt = [x.to(self.gpu_device) for x in batch]
                    out = self.net.forward(vec, sca, msk)

                    lcl_loss = linear_combination_loss(out, tgt, weight=wgt, **loss_kwargs)
                    pm_loss = abs_invariant_mass_squared_prediction_loss(out, weight=wgt)
                    total_loss = a_val * lcl_loss + b_val * pm_loss

                    for j, (name, coef) in enumerate(zip(base_loss_names, coeffs)):
                        comp_val_sum[j] += (
                            coef
                            * loss_function[name](out[:, j], tgt[:, j], torch.squeeze(wgt))
                            .detach()
                        )

                    epoch_val_loss.append(total_loss.detach())
                    pm_val_sum += pm_loss.detach()
                    aLCL_val_sum += (a_val * lcl_loss).detach()
                    bPM_val_sum += (b_val * pm_loss).detach()

            n_val_batches = len(self.validloader)
            self.valid_loss_list_per_epoch.append(torch.mean(torch.tensor(epoch_val_loss)))
            self.valid_comp_loss_per_epoch.append(comp_val_sum / n_val_batches)
            self.valid_pred_mass_per_epoch.append(pm_val_sum / n_val_batches)
            self.valid_alpha_LCL_per_epoch.append(aLCL_val_sum / n_val_batches)
            self.valid_beta_PM_per_epoch.append(bPM_val_sum / n_val_batches)

            # ------------------- checkpoint & plots -------------------
            self._checkpoint_and_plot(epoch, scheme="Hybrid", n_vars=n_vars)

        if self.logger:
            self.logger.close()

    # ------------------------------------------------------------------
    # helper – model checkpoint + plotting
    # ------------------------------------------------------------------

    def _checkpoint_and_plot(self, epoch: int, scheme: str, n_vars: int):
        # stdout
        print(
            f"Epoch: {epoch}, train_loss: {self.train_loss_list_per_epoch[-1]:0.4f}, "
            f"valid_loss: {self.valid_loss_list_per_epoch[-1]:0.4f}"
        )

        # best model tracking
        if self.valid_loss_list_per_epoch[-1] < self.bestscore:
            self.bestscore = self.valid_loss_list_per_epoch[-1]
            torch.save(self.net.state_dict(), os.path.join(self.save_path, "best_model.zip"))
            print(f"model saved with best score: {self.bestscore:0.4f}")

        # prepare metrics dict for plotting
        metrics: Dict[str, List] = {
            "Training Loss Total": self.train_loss_list_per_epoch,
            "Validation Loss Total": self.valid_loss_list_per_epoch,
        }

        # component curves
        for j in range(n_vars):
            metrics[f"Train: Var. {j}"] = [x[j].item() for x in self.train_comp_loss_per_epoch]
            #metrics[f"valid_var_{j}"] = [x[j].item() for x in self.valid_comp_loss_per_epoch]

        if scheme == "Hybrid":
            metrics.update(
                {
                    "$\\alpha$": self.alpha_list,
                    "$\\beta$": self.beta_list,
                    "Train: Pred. $m_{\\nu}^{2}$": self.train_pred_mass_per_epoch,
                    #"valid_predInvMass": self.valid_pred_mass_per_epoch,
                    "Train: $\\alpha \\cdot $LCL($p_{\\mu}$)": self.train_alpha_LCL_per_epoch,
                    #"valid_alpha*LCL": self.valid_alpha_LCL_per_epoch,
                    "Train: $\\beta \\cdot m_{\\nu}^{2}$": self.train_beta_PM_per_epoch,
                    #"valid_beta*PM": self.valid_beta_PM_per_epoch,
                }
            )

        # one canvas – many lines
        plot_loss(metrics, self.save_path)

    # ------------------------------------------------------------------
    # eval (unchanged)
    # ------------------------------------------------------------------

    def eval(self):
        self.net.to(self.gpu_device)
        self.net.load_state_dict(
            torch.load(os.path.join(self.save_path, "best_model.zip"), map_location=torch.device("cpu"))
        )
        self.net.eval()

        trueval, prediction = [], []
        # Make sure eval does not build graphs and uses lean CUDA paths
        with torch.inference_mode():
            for batch in self.testloader:
                vec, sca, msk, tgt = [x.to(self.gpu_device) for x in batch[:4]]
                out = self.net.forward(vec, sca, msk)
                trueval.append(tgt.detach().cpu().numpy())
                prediction.append(out.detach().cpu().numpy())

        np.savez(os.path.join(self.save_path, "result.npz"), trueval=np.concatenate(trueval), prediction=np.concatenate(prediction))

        # save true and predicted target variables to df 
        true_arr = np.concatenate(trueval, axis=0)
        pred_arr = np.concatenate(prediction, axis=0)
        
        target = self.input_d["target"]
        truevars = [f"true_{x}" for x in target]
        predvars = [f"pred_{x}" for x in target]
        
        truevars_df = pd.DataFrame(true_arr, columns=truevars)
        predvars_df = pd.DataFrame(pred_arr, columns=predvars)
        df = pd.concat([truevars_df, predvars_df], axis=1)
        
        # add tag column
        model_phys_name = self.input_d["model_phys_name"]
        df[model_phys_name] = -999999
        
        df.to_csv(os.path.join(self.save_path, "result.csv"), index=False)

        