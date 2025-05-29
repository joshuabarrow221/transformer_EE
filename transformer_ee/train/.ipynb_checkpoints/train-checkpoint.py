"""
A module for training a model.
"""

import json
import os
import math

import numpy as np
import torch

from transformer_ee.dataloader.load import get_train_valid_test_dataloader
from transformer_ee.logger.train_logger import BaseLogger
from transformer_ee.model import create_model
from transformer_ee.utils import get_gpu, hash_dict

from .loss import linear_combination_loss
from .loss import abs_invariant_mass_squared_prediction_loss
from .loss_track import plot_loss
from .optimizer import create_optimizer


"""
Decreasing S-curve from a_start -> a_end over epoch in [0,100].
Example: alpha(i, a_start=1, a_end=0.5) falls from 1 down toward 0.5.
"""
def alpha(epoch: float,
    a_start: float = 1.0,
    a_end:   float = 0.1,
    k:       float = 1.00,
    x0:      float = 6.0) -> float:
    return a_end + (a_start - a_end) / (1 + math.exp(k * (epoch - x0)))
"""
Increasing S-curve from b_start -> b_end over epoch in [0,100].
Example: beta(i, b_start=0, b_end=0.8) rises from 0 up toward 0.8.
"""
def beta(epoch: float,
    b_start: float = 0.0,
    b_end:   float = 0.9,
    k:       float = 1.00,
    x0:      float = 6.0) -> float:
    return b_start + (b_end - b_start) / (1 + math.exp(-k * (epoch - x0)))

class MVtrainer:
    r"""
    A class to train a model.

    You may want to load config from a json file. For example:
    ```

    with open("transformer_ee/config/input.json", encoding="UTF-8", mode="r") as f:
        input_d = json.load(f)

    ```
    """

    def __init__(self, input_d: dict, logger: BaseLogger | None = None):
        self.gpu_device = get_gpu()  # get gpu device
        self.input_d = input_d
        self.logger = logger
        print(json.dumps(self.input_d, indent=4))

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

        self.train_loss_list_per_epoch = (
            []
        )  # list to store training losses after each epoch
        self.valid_loss_list_per_epoch = (
            []
        )  # list to store validation losses after each epoch

        self.epochs = input_d["model"].get("epochs", 100)
        print("epochs: ", self.epochs)
        self.bestscore = 1e9
        self.print_interval = input_d.get("print_interval", 1)

        self.save_path = os.path.join(
            input_d.get("save_path", "."), "model_" + hash_dict(self.input_d)
        )
        os.makedirs(
            self.save_path,
            exist_ok=True,
        )
        input_json = os.path.join(self.save_path, "input.json")
        if not os.path.exists(input_json):
            with open(input_json, "w") as f:
                json.dump(self.input_d, f, indent=4)

        stat_json = os.path.join(self.save_path, "trainset_stat.json")
        if not os.path.exists(stat_json):
            with open(stat_json, "w") as f:
                json.dump(self.train_set_stat, f, indent=4)

    def train(self):
        r"""
        Train the model.
        """
        
        for i in range(self.epochs):
            self.net.train()  # begin training

            batch_train_loss = []

            for batch_idx, batch in enumerate(self.trainloader):

                vector_train_batch = batch[0].to(
                    self.gpu_device
                )  # shape: (batch_size, max_seq_len, vector_dim)
                scalar_train_batch = batch[1].to(
                    self.gpu_device
                )  # shape: (batch_size, scalar_dim)
                mask_train_batch = batch[2].to(
                    self.gpu_device
                )  # shape: (batch_size, max_seq_len)
                target_train_batch = batch[3].to(
                    self.gpu_device
                )  # shape: (batch_size, target_dim)
                weight_train_batch = batch[4].to(
                    self.gpu_device
                )  # shape: (batch_size, 1)
                Netout = self.net.forward(
                    vector_train_batch, scalar_train_batch, mask_train_batch
                )
                # This will call the forward function, usually it returns tensors.

                #loss = linear_combination_loss( Netout,target_train_batch,weight=weight_train_batch,**self.input_d["loss"]["kwargs"], )  # regression loss
                loss = alpha(i)*linear_combination_loss( Netout,target_train_batch,weight=weight_train_batch,**self.input_d["loss"]["kwargs"], ) + beta(i)*abs_invariant_mass_squared_prediction_loss( Netout,weight=weight_train_batch, )
                # Zero the gradients before running the backward pass.
                self.optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to all the learnable
                # parameters of the model. Internally, the parameters of each Module are stored
                # in Tensors with requires_grad=True, so this call will compute gradients for
                # all learnable parameters in the model.
                loss.backward()
                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self.optimizer.step()

                batch_train_loss.append(loss)
                if batch_idx % self.print_interval == 0:
                    print(
                        "Epoch: {}, batch: {} Loss: {:0.4f}".format(i, batch_idx, loss)
                    )
                if self.logger is not None:
                    self.logger.log_scalar(
                        scalars={
                            "train/loss": loss.item(),
                            "train/step": i * len(self.trainloader) + batch_idx,
                        },
                        step=i * len(self.trainloader) + batch_idx,
                        epoch=i,
                    )
            self.train_loss_list_per_epoch.append(
                torch.mean(torch.Tensor(batch_train_loss))
            )

            self.net.eval()  # begin validation

            # self.net.to("cpu")

            batch_valid_loss = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.validloader):
                    vector_valid_batch = batch[0].to(self.gpu_device)
                    scalar_valid_batch = batch[1].to(self.gpu_device)
                    mask_valid_batch = batch[2].to(self.gpu_device)
                    target_valid_batch = batch[3].to(self.gpu_device)
                    weight_valid_batch = batch[4].to(self.gpu_device)

                    Netout = self.net.forward(
                        vector_valid_batch, scalar_valid_batch, mask_valid_batch
                    )
                    # This will call the forward function, usually it returns tensors.

                    #loss = linear_combination_loss( Netout,target_valid_batch,weight=weight_valid_batch,**self.input_d["loss"]["kwargs"], )
                    #print(weight_valid_batch)
                    loss = alpha(i)*linear_combination_loss( Netout,target_valid_batch,weight=weight_valid_batch,**self.input_d["loss"]["kwargs"], ) + beta(i)*abs_invariant_mass_squared_prediction_loss( Netout,weight=None, )
                

                    batch_valid_loss.append(loss)
                    if batch_idx % self.print_interval == 0:
                        print(
                            "Epoch: {}, batch: {} Loss: {:0.4f}".format(
                                i, batch_idx, loss
                            )
                        )
                    if self.logger is not None:
                        self.logger.log_scalar(
                            scalars={
                                "val/loss": loss.item(),
                                "val/step": i * len(self.trainloader) + batch_idx,
                            },
                            step=i * len(self.trainloader) + batch_idx,
                            epoch=i,
                        )
                self.valid_loss_list_per_epoch.append(
                    torch.mean(torch.Tensor(batch_valid_loss))
                )

            # self.net.to(self.gpu_device)

            print(
                "Epoch: {}, train_loss: {:0.4f}, valid_loss: {:0.4f}".format(
                    i,
                    self.train_loss_list_per_epoch[-1],
                    self.valid_loss_list_per_epoch[-1],
                )
            )

            if self.valid_loss_list_per_epoch[-1] < self.bestscore:
                self.bestscore = self.valid_loss_list_per_epoch[-1]
                torch.save(
                    self.net.state_dict(),
                    os.path.join(self.save_path, "best_model.zip"),
                )
                print("model saved with best score: {:0.4f}".format(self.bestscore))

            plot_loss(
                self.train_loss_list_per_epoch,
                self.valid_loss_list_per_epoch,
                self.save_path,
            )
        if self.logger is not None:
            self.logger.close()

    def eval(self):
        r"""
        Evaluate the model.
        """
        self.net.load_state_dict(
            torch.load(
                os.path.join(self.save_path, "best_model.zip"),
                map_location=torch.device("cpu"),
            )
        )
        self.net.eval()
        self.net.to(self.gpu_device)

        trueval = []
        prediction = []

        for _batch_idx, batch in enumerate(self.testloader):
            vector_valid_batch = batch[0].to(self.gpu_device)
            scalar_valid_batch = batch[1].to(self.gpu_device)
            mask_valid_batch = batch[2].to(self.gpu_device)
            target_valid_batch = batch[3].to(self.gpu_device)

            Netout = self.net.forward(
                vector_valid_batch, scalar_valid_batch, mask_valid_batch
            )
            trueval.append(target_valid_batch.detach().cpu().numpy())
            prediction.append((Netout.detach().cpu().numpy()))

        trueval = np.concatenate(trueval)  # [:,0]
        prediction = np.concatenate(prediction)
        np.savez(
            os.path.join(self.save_path, "result.npz"),
            trueval=trueval,
            prediction=prediction,
        )