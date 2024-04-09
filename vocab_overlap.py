#  Calculate vocabulary overlap.

from probe.probe import get_original_word_representations, convert_to_ids
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

def get_vocab_ids(sent_list, tokenizer):
    vocab_ids = set()
    for sent in sent_list:
        words = [w.form for w in sent]
        _, ids = convert_to_ids(words, tokenizer=tokenizer)
        ids = ids.tolist()
        vocab_ids = vocab_ids.union(*ids)
    return vocab_ids

if __name__ == "__main__":
    # Set up argument parser.
    parser = argparse.ArgumentParser("")
    parser.add_argument("--data_dir")
    parser.add_argument("--pretrained")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    
    out_dir = args.out_dir #"results/vocab_overlap.tsv"
    data_dir = args.data_dir # "preprocessed"
    pretrained = args.pretrained#"google-bert/bert-base-multilingual-cased"

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    langs = os.listdir(data_dir)
    langs_ids = dict()


    for language in langs:
        print("Processing language {}".format(language))
        path = os.path.join(data_dir, language)
        data = load_json(os.path.join(path, "preprocessed.json"))
        train_sents = data.train()
        ids = get_vocab_ids(train_sents, tokenizer=tokenizer) # Get set of vocabulary ids.
        langs_ids[language] = ids
    
    overlap = dict()
    # Filter for languages with training material.
    train_langs = [lang for lang in langs_ids if len(langs_ids[lang]) != 0]

    for lang_i in tqdm(train_langs):
        overlap[lang_i] = []
        ids_i = langs_ids[lang_i]
        for lang_j in train_langs:
            ids_j = langs_ids[lang_j]
            intersect = ids_i.intersection(ids_j) # Compute set of ids that are present in both languages.
            prop_common = len(intersect) / len(ids_i) # Compute proportion in lang_i
            overlap[lang_i].append(prop_common)

    # Save tsv file.
    df = pd.DataFrame.from_dict(overlap, orient="index", columns=train_langs)
    df.to_csv(out_dir, sep="\t")

    

