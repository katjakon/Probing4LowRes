# Calculate evennes in data set.
import argparse
from collections import Counter
import json
import math
import os

import pandas as pd

from utils.read_conllu import Data

def load_json(path):
    with open(path, encoding="utf-8") as file:
        split_dict = json.load(file)
    return Data.from_json(split_dict)

if __name__ == "__main__":
    # Set up argument parser.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--property")
    args = parser.parse_args()

    data_dir = args.data_dir
    property = args.property
    out_dir = args.out_dir

    langs = os.listdir(data_dir)
    data_df = {"Language": [], "Shannon Evenness Index": [], "#Labels": []}
    for language in langs:
        print("Processing language {}".format(language))
        path = os.path.join(data_dir, language)
        data = load_json(os.path.join(path, "preprocessed.json"))
        train_sents = data.train()

        if len(train_sents) == 0:
                continue

        label_counter = Counter()

        for idx, sent in enumerate(train_sents):
            for word in sent:
                if property == "upos":
                    label = word.upos
                else:
                    feats = word.feats
                    label = feats.get(property)
                if label is not None:
                        label_counter[label] += 1
        # Maximum Shannon Entropy
        num_labels = len(label_counter.keys())
        if num_labels <= 1:
             continue
        
        max_entropy = math.log(num_labels)
        # Actual Shannon Entropy
        total_count = label_counter.total()
        H = 0
        for label, count in label_counter.items():
                prop_label = count / total_count
                information = prop_label * math.log(prop_label)
                H += information
        H = -1 * H
        # Shannon Evennes Index
        evenness_index = H / max_entropy

        # Add results fo dictionary.
        data_df["Language"].append(language)
        data_df["#Labels"].append(num_labels)
        data_df["Shannon Evenness Index"].append(evenness_index)
        print("Evenness Index: ", round(evenness_index, ndigits=3)) 
        print("Number of labels: ", num_labels)
        print("===="*30)
    
    df = pd.DataFrame(data_df)
    df.to_csv(out_dir, sep="\t")    