from wsi_mil.deepmil.train import main as train
from wsi_mil.deepmil.writes_results_cross_val import main as writes_validation_results
from wsi_mil.deepmil.writes_final_results import main as writes_test_results
import pandas as pd
import os
import datetime
import shutil
from argparse import ArgumentParser
import yaml
import torch
from trackers import timetracker

# For the sklearn warnings
import warnings

warnings.filterwarnings("ignore")

print("Working in conda env = " + os.environ["CONDA_PREFIX"])
print(f"Cuda is available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device = {torch.cuda.get_device_name(0)}")
else:
    print("Device = cpu")

parser = ArgumentParser()
parser.add_argument("--out", type=str, help="Path where will be stored the outputs")
parser.add_argument(
    "--name",
    type=str,
    help="Name of the experiment, current date by default",
    default=None,
)
parser.add_argument("--rep", type=int, help="Number of repetitions for each test sets.")
parser.add_argument("--config", type=str, help="Path to the config file.")
parser.add_argument(
    "--n_ensemble",
    type=int,
    help="Number of model to ensemble. selected wrt validation results.",
    default=1,
)
args = parser.parse_args()
timer = timetracker(name="tracker", verbose=1)
if args.name is None:
    args.name = datetime.date.today().strftime("%Y_%m_%d")
out = os.path.abspath(args.out)
out = os.path.join(out, args.name)
if not os.path.exists(out):
    os.makedirs(out)
args.config = os.path.abspath(args.config)
os.makedirs(out, exist_ok=True)
shutil.copy(args.config, os.path.join(out, "config.yaml"))
with open(args.config, "r") as f:
    dic = yaml.safe_load(f)
table = pd.read_csv(dic["table_data"])
tests = len(set(table["test"].values))
if "nb_tiles" in dic:
    nb_tiles = dic["nb_tiles"]
    print(f"will train with nb_tiles = {nb_tiles}")
timer.tic()
for test in range(tests):
    for rep in range(args.rep):
        print(f"Working on: rep={rep+1}/{args.rep}, fold={test+1}/{tests}")
        raw_args = [
            "--config",
            args.config,
            "--repeat",
            f"{rep}",
            "--test_fold",
            f"{test}",
        ]
        wd = os.path.join(out, f"test_{test}", f"rep_{rep}")
        os.makedirs(wd, exist_ok=True)
        os.chdir(wd)
        train(raw_args=raw_args)
timer.toc()
# Root of experiment.
os.chdir(out)
raw_args = ["--n_ensemble", f"{args.n_ensemble}"]
writes_validation_results(raw_args)
writes_test_results([])
print(f"Done!")
