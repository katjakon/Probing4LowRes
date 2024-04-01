from typing import List
import os
import random

class Data:

    def __init__(self, path: str=None, split_dict=None) -> None:
        self.path = path # Directory to conllu files
        if self.path is not None:
            self.splits = [os.path.join(self.path, split) 
                        for split in os.listdir(self.path)]
        if split_dict is not None:
            self.splits = split_dict
        assert path is not None or split_dict is not None 
    
    def read_conllu(self, path: str, limit=None):
        sents = []
        with open(path, encoding="utf-8") as file:
            sent = []
            for line in file:
                if line.strip():
                    if not line.startswith("#"):
                        sent.append(line.strip())
                else: # Empty line means a new sentence will start.
                    if limit is not None and len(sents) >= limit:
                        break
                    sent_object = Sentence(sent)
                    sents.append(sent_object)
                    sent = []
        return sents
    
    def features(self, sents):
        feat_set = dict()
        for sent in sents:
            for word in sent:
                pos = word.upos
                feats = word.feats
                if pos not in feat_set:
                    feat_set[pos] = dict()
                for f in feats:
                    if f not in feat_set[pos]:
                        feat_set[pos][f] = set()
                    feat_set[pos][f].add(feats[f])
        return feat_set
    
    def feature_coverage(self, sents):
        totals = dict()
        feat_total = dict()
        val_total = dict()
        # Count up features and pos tags
        for sent in sents:
            for word in sent:
                pos = word.upos
                feats = word.feats
                if pos not in totals:
                    totals[pos] = 0
                    feat_total[pos] = dict()
                    val_total[pos] = dict()
                totals[pos] += 1
                for f in feats:
                    if f not in feat_total[pos]:
                        feat_total[pos][f] = 0
                        val_total[pos][f] = dict()
                    feat_total[pos][f] += 1
                    feat_val = feats[f]
                    if feat_val not in val_total[pos][f]:
                        val_total[pos][f][feat_val] = 0
                    val_total[pos][f][feat_val] += 1
        # Calculate proportion of features in individual pos tags
        prop_feats = dict()
        for pos in totals:
            prop_feats[pos] = dict()
            total_count = totals[pos]
            for feat in feat_total[pos]:
                count_feat = feat_total[pos][feat]
                prop_feats[pos][feat] = count_feat / total_count
        # Calculate proportion of values for each feature
        val_prop = dict()
        for pos in val_total:
            val_prop[pos] = dict()
            for feat in val_total[pos]:
                if feat not in val_prop[pos]:
                    val_prop[pos][feat] = dict()
                feat_count = feat_total[pos][feat]
                for val in val_total[pos][feat]:
                    val_prop[pos][feat][val] = val_total[pos][feat][val] / feat_count
        # Calculate proportion of tags
        sum_pos = sum(totals.values())
        prop_pos = {pos: totals[pos] /  sum_pos for pos in totals}
        return prop_pos , prop_feats, val_prop
    
    def train(self, verbose=False, sample_size=None, seed=None, limit=None, strict_sample=True):
        random.seed(seed)
        train_data = []
        if isinstance(self.splits, list):
            for split in self:
                if "train" in split:
                    if verbose:
                        print("Getting data from split {}".format(split))
                    sents = self.read_conllu(split, limit=limit)
                    train_data.extend(sents)
        if isinstance(self.splits, dict):
            train_data = self.splits.get("train", [])
        if sample_size is None:
            return train_data
        else:
            if strict_sample is True:
                return random.sample(train_data, k=sample_size)
            else: 
                return train_data
    
    def test(self, verbose=False):
        test_data = []
        if isinstance(self.splits, list):
            for split in self:
                if "test" in split:
                    if verbose:
                        print("Getting data from split {}".format(split))
                    sents = self.read_conllu(split)
                    test_data.extend(sents)
        if isinstance(self.splits, dict):
            test_data = self.splits.get("test", [])
        return test_data
    
    def dev(self, verbose=False):
        val_data = []
        if isinstance(self.splits, list):
            for split in self:
                if "dev" in split:
                    if verbose:
                        print("Getting data from split {}".format(split))
                    sents = self.read_conllu(split)
                    val_data.extend(sents)
        if isinstance(self.splits, dict):
            val_data = self.splits.get("dev", [])
        return val_data

    def __iter__(self):
        return iter(self.splits)

    def __repr__(self) -> str:
        return "Data(dir={})".format(self.path)

    def to_json(self, train_sample_size=None, seed=None):
        train = [s.to_json() for s in self.train(
            sample_size=train_sample_size,
            seed=seed)]
        test = [s.to_json() for s in self.test()]
        dev = [s.to_json() for s in self.dev()]
        return {
            "train": train, 
            "test": test,
            "dev": dev
        }
    
    @classmethod
    def from_json(cls, json_dict):
        split_dict = dict()
        for key in json_dict:
            val = json_dict[key]
            split_dict[key] = [Sentence(word_list) for word_list in val]
        return cls(
            split_dict=split_dict
        )


class Sentence:

    def __init__(self, word_list: List[List[str]]) -> None:
        self.word_list = word_list
        self.words = self.read_words(word_list)

    def read_words(self, word_list):
        words = []
        for word in word_list:
            w = Word(word.split("\t"))
            words.append(w)
        return words
    
    def __getitem__(self, index):
        return self.words[index]
    
    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __repr__(self) -> str:
        word_forms = [w.form for w in self.words]
        s = "Sentence(#words={},'{}' ... )".format(len(self), " ".join(word_forms[:3]))
        return s
    
    def to_json(self):
        return [w.to_string() for w in self]

class Word:

    INDEX = {
        "id": 0,
        "form": 1,
        "lemma": 2,
        "upos": 3,
        "xpos": 4,
        "feats": 5
    }

    def __init__(self, word_feats: List[str]) -> None:
        self.word_feats = word_feats
        self._feats = None
    
    @property
    def form(self):
        return self.word_feats[self.INDEX["form"]]
    
    @property
    def id(self):
        return self.word_feats[self.INDEX["id"]]

    @property
    def lemma(self):
        return self.word_feats[self.INDEX["lemma"]]
    
    @property
    def upos(self):
        return self.word_feats[self.INDEX["upos"]]

    @property
    def feats(self):
        if self._feats is not None:
            return self._feats
        feats = self.word_feats[self.INDEX["feats"]]
        feat_dict = dict()
        if feats == "_":
            return feat_dict
        feats = feats.split("|")
        for f in feats:
            feat_key, feat_value = f.split("=")
            feat_dict[feat_key] = feat_value
        self._feats = feat_dict
        return self._feats

    def __repr__(self) -> str:
        return "Word({} {} {})".format(self.id, self.form, self.upos)
    
    @classmethod
    def from_string(cls, string, sep="\t"):
        ls = string.split(sep)
        return cls(ls)

    def to_string(self, sep="\t"):
        string = sep.join(self.word_feats)
        return string

if __name__ == "__main__":
    path = "data/Tamil"
    langs = os.listdir("data")
    for l in langs:
        path = os.path.join("data", l)
        data = Data(path)
        train = data.train()
        n_sents = len(train)
        if n_sents < 100:
            print(l, n_sents)
    # data = Data(path)
    # for file in data.splits:
    #     if "train" in file:
    #         pos_prop, feats_props, val_props = data.feature_coverage(split=file)
    #         feats = data.features(split=file)
    #         for pos in feats_props:
    #             print("{} ({}%)".format(pos, round(100*pos_prop[pos], ndigits=2)))
    #             for feat in feats_props[pos]:
    #                 val_str = ""
    #                 for val in val_props[pos][feat]:
    #                     s = "{} ({}%) ".format(
    #                         val,
    #                         round(100*val_props[pos][feat][val], ndigits=2)
    #                     )
    #                     val_str += s
    #                 print("{}: {}%\tvalues: {}".format(
    #                     feat,
    #                     round(feats_props [pos][feat]*100, ndigits=2),
    #                     val_str)
    #                     )
    #             print("--"*30)