## Version of train_script.py developed by Leon Tong that accepts command line inputs for output directory (user should adjust the base directory to suit their needs) and whether or not to use the WandB logger and funnels standard output to a log file.

import json
import sys
import os

from transformer_ee.train import MVtrainer

with open("/home/ltong/transformer_EE/transformer_ee/config/input_nova_mprod6_1_OPAL_fd_rhc.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

## Set the path to the input data
# input_d["data_path"]="transformer_ee/data/dataset_lstm_ee_nd_rhc_nonswap_loose_cut.csv.xz"

## Set the number of workers in dataloader. The more workers, the faster the data loading. But it may consume more memory.
input_d["num_workers"] = 10

## Set the model hyperparameters
# input_d["model"]["kwargs"]["nhead"] = 12
input_d["model"]["epochs"] = 1000
input_d["model"]["stop_loss"] = 5
input_d["model"]["kwargs"]["num_layers"] = 5

## Set the optimizer
input_d["optimizer"]["name"] = "sgd"
input_d["optimizer"]["kwargs"]["lr"] = 0.01
input_d["optimizer"]["kwargs"]["momentum"] = 0.9

## Set the path to save the model
out_dir = sys.argv[1]
input_d["save_path"] = "/exp/nova/data/users/ltong/transformerEE/" + out_dir

## Save output to log file.
orig_stdout = sys.stdout
os.mkdir(input_d["save_path"])
f = open(input_d["save_path"]+"/train.log", 'a')
sys.stdout = f

## Set the weighter
# input_d["weight"] = {"name": "FlatSpectraWeights", "kwargs": {"maxweight": 5, "minweight": 0.2}}

input_d["dataframe_type"] = "polars"

## Example of adding noise to the input variables
# input_d["noise"] = {
#     "name": "gaussian",
#     "mean": 0,
#     "std": 0.2,
#     "vector": ["particle.energy", "particle.calE", "particle.nHit"],
#     "scalar": ["event.calE", "event.nHits"],
# }

# Example of using WandBLogger
if sys.argv[2] == "log":
    from transformer_ee.logger.wandb_train_logger import WandBLogger
    my_logger = WandBLogger(
        project="test", entity="neutrinoenenergyestimators", config=input_d, dir="save", id=out_dir
    )
    my_trainer = MVtrainer(input_d, logger=my_logger)

else:
    my_trainer = MVtrainer(input_d)

my_trainer.train()
my_trainer.eval()

sys.stdout = orig_stdout
f.close()
