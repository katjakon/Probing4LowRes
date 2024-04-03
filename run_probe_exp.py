import argparse
import json
import os

import numpy as np
import pandas as pd

from utils.read_conllu import Data
from probe.probe_classifiers import ClassifierProbe
from probe.majority_probe import MajorityBaseline
from probe.random_probe import RandomBaseline
from probe.control_task_probe import ControlTaskProbe

def load_json(path):
    with open(path, encoding="utf-8") as file:
        split_dict = json.load(file)
    return Data.from_json(split_dict)

parser = argparse.ArgumentParser("Run probing experiments.")
parser.add_argument("--preprocessed_dir")
parser.add_argument("--out_path")
parser.add_argument("--property")
parser.add_argument("--clf_type")

args = parser.parse_args()

preprocessed_dir = args.preprocessed_dir
langs = os.listdir(preprocessed_dir)
out = args.out_path
property = args.property
clf_type = args.clf_type

data_df = {
    "Language": [],
    "Classes": [],
    f"{clf_type} Accuracy": [],
    f"{clf_type} Balanced Accuracy": [],
    f"{clf_type} Sensitivity": [],
    "Majority Baseline Balanced Accuracy": [],
    "Random Baseline Balanced Accuracy": [],
    "Majority Baseline Accuracy": [],
    "Random Baseline Accuracy": [],
    }

smaller_sample = ""
for lang in langs:
    print("==="*30)
    print(f"Train probe for {lang}...")
    # Load data
    path = os.path.join(preprocessed_dir, lang)
    data = load_json(os.path.join(path, "preprocessed.json"))
    test_repr = np.load(os.path.join(path, "test.npz"))
    train_repr = np.load(os.path.join(path, "train.npz"))
    if len(train_repr) == 0:
        print(f"No train data found for {lang}")
        continue
    if len(train_repr) < 500:
        smaller_sample = "*"

    clf_probe = ClassifierProbe(data=data, train_repr=train_repr, test_repr=test_repr, clf_type=clf_type , property=property)
    majority_baseline = MajorityBaseline(data=data, property=property)
    random_baseline = RandomBaseline(data=data, train_repr=train_repr, test_repr=test_repr, clf_type=clf_type , property=property)
    control_probe = ControlTaskProbe(data=data, train_repr=train_repr, test_repr=test_repr, clf_type=clf_type , property=property)

    # Training 
    try:
        clf_probe.train() 
        classes = clf_probe.classes
        accs = clf_probe.evaluate()
        print(f"Classifier Probe: {accs}")
        ###
    except ValueError:
        print("Couldn't train.")
        continue

    data_df["Language"].append(lang)
    data_df["Classes"].append(",".join(classes))
    data_df[f"{clf_type} Accuracy"].append(accs["Accuracy"])
    data_df[f"{clf_type} Balanced Accuracy"].append(accs["Balanced Accuracy"])

    clf_acc = accs["Accuracy"]
    ###############################
    majority_baseline.train()
    accs = majority_baseline.evaluate()
    print(f"Majority Baseline: {accs}")
    data_df["Majority Baseline Accuracy"].append(accs["Accuracy"])
    data_df["Majority Baseline Balanced Accuracy"].append(accs["Balanced Accuracy"])

    ######################
    random_baseline.train()
    accs = random_baseline.evaluate()
    print(f"Random Baseline: {accs}")
    data_df["Random Baseline Accuracy"].append(accs["Accuracy"])
    data_df["Random Baseline Balanced Accuracy"].append(accs["Balanced Accuracy"])

    ##################
    control_probe.train()
    accs = control_probe.evaluate()
    print(f"Control Task Probe: {accs}")
    data_df[f"{clf_type} Sensitivity"].append(
        clf_acc - accs["Accuracy"]
    )

    smaller_sample = ""


df = pd.DataFrame(data_df)
df.to_csv(out, sep="\t")

