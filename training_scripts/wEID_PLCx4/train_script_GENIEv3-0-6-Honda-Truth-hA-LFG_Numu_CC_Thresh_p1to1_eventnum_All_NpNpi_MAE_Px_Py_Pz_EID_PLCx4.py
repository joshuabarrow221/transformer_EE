import json
import os
import argparse
from transformer_ee.train import MVtrainer
from transformer_ee.logger.wandb_train_logger import WandBLogger
import wandb
import shutil

with open("/home/jbarrow/Transformer/transformer_EE/transformer_ee/config/wEID_PLCx4/input_GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_Px_Py_Pz_EID_PLCx4.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

# # Set the path to the input data
input_d["data_path"]="/exp/dune/app/users/rrichi/FinalCodes/Numu_CC_Thresh_p1to1_VectorLeptWithoutNC_eventnum_All_NpNpi.csv"
#input_d["data_path"]="/exp/dune/app/users/rrichi/FinalCodes/Numu_CC_Thresh_p1to1_VectorLeptWithoutNC_eventnum_All_1p0pi.csv"

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
input_d["save_path"] = "/exp/dune/data/users/jbarrow/save_genie-TEST/model/test/GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_Px_Py_Pz_EID_PLCx4"
#input_d["save_path"] = "/home/cborden/git/richi_transformer_EE/save/model/test"

input_d["dataframe_type"] = "polars"

my_logger = WandBLogger(project="GENIE_Atmo",entity="neutrinoenenergyestimators", config=input_d, dir="/exp/dune/data/users/jbarrow/save_genie-TEST/wandb", id="GENIEv3-0-6-Honda-Truth-hA-LFG_Numu_CC_Thresh_p1to1_eventnum_All_NpNpi_MAE_Px_Py_Pz_EID_PLCx4")
my_trainer = MVtrainer(input_d, logger=my_logger)
#my_trainer = MVtrainer(input_d)

my_trainer.train_LCL()
#my_trainer.train_LCL_wPredInvMass()
my_trainer.eval()
