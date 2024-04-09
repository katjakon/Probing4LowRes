#  Calculate vocabulary overlap.

from probe.probe import convert_to_ids
from utils.read_conllu import Data

import argparse
import json
import pandas as pd
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def load_json(path):
    with open(path, encoding="utf-8") as file:
        split_dict = json.load(file)
    return Data.from_json(split_dict)

def get_prop_unk(sent_list, tokenizer, tok_id):
    unk_counter = 0
    token_count = 0
    for sent in sent_list:
        words = [w.form for w in sent]
        _, ids = convert_to_ids(words, tokenizer=tokenizer)
        token_count += len(ids[0])
        for i in ids[0]:
            if i == tok_id:
                unk_counter += 1
    return unk_counter / token_count

if __name__ == "__main__":
    # Set up argument parser.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir")
    parser.add_argument("--pretrained")
    parser.add_argument("--out_dir")
    parser.add_argument("--unk_token_id", default=100)
    args = parser.parse_args()
    unk_token_id = args.unk_token_id
    out_dir = args.out_dir #"results/vocab_overlap.tsv"
    data_dir = args.data_dir # "preprocessed"
    pretrained = args.pretrained#"google-bert/bert-base-multilingual-cased"

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    langs = os.listdir(data_dir)
    langs_ids = dict()

    data_dict = {"Language": [], "UNK Prop": []}
    for language in langs:
        print("Processing language {}".format(language))
        path = os.path.join(data_dir, language)
        data = load_json(os.path.join(path, "preprocessed.json"))
        train_sents = data.train()
        if len(train_sents) == 0:
            continue
        prop_unk = get_prop_unk(train_sents, tokenizer, tok_id=unk_token_id)
        data_dict["Language"].append(language)
        data_dict["UNK Prop"].append(prop_unk)
    
    df = pd.DataFrame(data_dict)
    df.to_csv(out_dir, sep="\t")
    

