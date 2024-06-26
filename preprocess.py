import argparse
import json
import numpy as np
import os
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

from probe.probe import get_original_word_representations, convert_to_ids
from utils.read_conllu import Data


def convert(sent_list, tokenizer, model):
    json_repr = []
    bert_repr = []
    for sent in sent_list:
        words = [w.form for w in sent]
        json_repr.append(sent.to_json())
        mapping, ids = convert_to_ids(words, tokenizer=tokenizer)
        repr = get_original_word_representations(ids, model, mapping)
        bert_repr.append(repr.detach().numpy())
    return json_repr, bert_repr

if __name__ == "__main__":
    # Set up argument parser.
    parser = argparse.ArgumentParser("Convert and sample data.")
    parser.add_argument("--data_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--pretrained")
    parser.add_argument("--out_dir")
    parser.add_argument('--strict_sample',
                    action='store_true') 
    args = parser.parse_args()

    seed = args.seed
    sample_size = args.sample_size
    PRETRAINED_MODEL = args.pretrained #"google-bert/bert-base-multilingual-cased"
    data_dir = args.data_dir
    out_dir = args.out_dir
    strict_sample = args.strict_sample

    # Load pretrained model.
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = BertModel.from_pretrained(PRETRAINED_MODEL)
    

    # Start converting.
    langs = os.listdir(data_dir)
    progress_bar = tqdm(langs)
    for lang in progress_bar:
        progress_bar.set_description(f"Processing {lang}")
        data_path = os.path.join(data_dir, lang)
        out_path = os.path.join(out_dir, lang)
        if os.path.exists(out_path):
            print(f"Skipped {lang},  {out_path} already exists.")
            continue
        data = Data(data_path)
        # Convert train
        try:
            train = data.train(
                sample_size=sample_size,
                seed=seed,
                strict_sample=strict_sample
            )
        except ValueError:
            print(f"Skipped {lang}, not enough samples in train.")
            continue

        train_json, train_repr = convert(train, tokenizer=tokenizer, model=model)

        # Convert test
        test = data.test()
        test_json, test_repr = convert(test, tokenizer=tokenizer, model=model)    

        # Convert dev
        dev = data.dev()
        dev_json, dev_repr = convert(dev, tokenizer=tokenizer, model=model)  

        # Create output dir.
        out_path = os.path.join(out_dir, lang)
        os.mkdir(out_path)

        # Save json dict
        json_dict = {
            "train": train_json,
            "dev": dev_json,
            "test": test_json
        }

        with open(os.path.join(out_path, "preprocessed.json"), "w", encoding="utf-8") as json_f:
            json.dump(json_dict, json_f)

        np.savez_compressed(
            os.path.join(out_path, "test"),
            **{str(idx): sent for idx, sent in enumerate(test_repr)}
        )
        np.savez_compressed(
            os.path.join(out_path, "train"),
            **{str(idx): sent for idx, sent in enumerate(train_repr)}
        )
        np.savez_compressed(
            os.path.join(out_path, "dev"),
            **{str(idx): sent for idx, sent in enumerate(dev_repr)}
        )
