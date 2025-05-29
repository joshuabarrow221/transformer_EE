import json
import os
import argparse
from transformer_ee.train import MVtrainer
from transformer_ee.logger.wandb_train_logger import WandBLogger
import wandb
import shutil

with open("/home/cborden/git/richi_transformer_EE/transformer_EE/transformer_ee/config/input_GENIEv3-4-0-flat-beam-spectra-Truth-Ar23_Numu_Inclusive_Thresh_p1to10_eventnum_100000_NpNpi_MAE-TEST.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

# # Set the path to the input data
input_d["data_path"]="/exp/dune/app/users/rrichi/FinalCodes/AnyNu_Inclusive_Thresh_0to1p2_VectorLeptw0NC_eventnum_All_NpNpi.csv"

## Set the number of workers in dataloader. The more workers, the faster the data loading. But it may consume more memory.
input_d["num_workers"] = 10

## Set the model hyperparameters
input_d["model"]["kwargs"]["nhead"] = 2
input_d["model"]["epochs"] = 20
input_d["model"]["kwargs"]["num_layers"] = 5

## Set the optimizer
input_d["optimizer"]["name"] = "sgd"
input_d["optimizer"]["kwargs"]["lr"] = 0.01
input_d["optimizer"]["kwargs"]["momentum"] = 0.9

## Set the path to save the model
input_d["save_path"] = "/exp/dune/data/users/cborden/save_genie-TEST/save/model/test"
#input_d["save_path"] = "/home/cborden/git/richi_transformer_EE/save/model/test"

input_d["dataframe_type"] = "polars"

# my_trainer = MVtrainer(input_d, logger=my_logger)
my_logger = WandBLogger(project="GENIE-Beam",entity="neutrinoenenergyestimators", config=input_d, dir="/exp/dune/data/users/cborden/save_genie-TEST", id="testrun_alpha_1_0p1_1_6__beta_0_0p9_1_6")
my_trainer = MVtrainer(input_d, logger=my_logger)
#my_trainer = MVtrainer(input_d)

my_trainer.train()
my_trainer.eval()
