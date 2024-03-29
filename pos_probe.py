import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils.read_conllu import Data
from probe.probe import convert_to_ids, get_original_word_representations


def features_and_labels(split, bert_npz):
    features = []
    labels = []
    for idx, sent in enumerate(split):
        repr = bert_npz[str(idx)]
        # Truncate tags
        tags = [w.upos for w in sent][:repr.shape[0]]
        features.extend(repr)
        labels.extend(tags)
    return features, labels

def load_json(path):
    with open(path, encoding="utf-8") as file:
        split_dict = json.load(file)
    return Data.from_json(split_dict)


preprocessed_dir = "preprocessed"
langs = os.listdir(preprocessed_dir)
out = "pos_acc.csv"

data_df = {"language": [], "accuracy": []}

for lang in langs:
    print(f"Train probe for {lang}...", end="")
    # Load data
    path = os.path.join(preprocessed_dir, lang)
    data = load_json(os.path.join(path, "preprocessed.json"))
    test_repr = np.load(os.path.join(path, "test.npz"))
    train_repr = np.load(os.path.join(path, "train.npz"))
    train_features, train_labels = features_and_labels(data.train(), train_repr)
    test_features, test_labels = features_and_labels(data.test(), test_repr)
    # Train classifier
    clf = SGDClassifier()
    clf.fit(train_features, train_labels)
    # Predict on test
    preds = clf.predict(test_features)
    acc = accuracy_score(y_true=test_labels, y_pred=preds)
    # Save results
    data_df["language"].append(lang)
    data_df["accuracy"].append(acc)
    print(f"Accuracy: {acc}")

df = pd.DataFrame(data_df)
df.to_csv(out, index=False)
print("Latex table:")
print(df.to_latex(index=False))





