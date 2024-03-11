from typing import List
import os
class Data:

    def __init__(self, path: str) -> None:
        self.path = path # Path to conllu file.
        self._splits = os.listdir(self.path)
        self.splits = dict()
        for split in self._splits:
            split_path = os.path.join(self.path, split)
            self.splits[split_path] = self.read_conllu(split_path)
    
    def read_conllu(self, path: str):
        sents = []
        with open(path, encoding="utf-8") as file:
            sent = []
            for line in file:
                if line.strip():
                    if not line.startswith("#"):
                        sent.append(line.strip())
                else: # Empty line means a new sentence will start.
                    sent_object = Sentence(sent)
                    sents.append(sent_object)
                    sent = []
        return sents
    
    def features(self):
        feat_set = dict()
        for sent in self.sents:
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

    def __iter__(self):
        return iter(self.splits)

    def __repr__(self) -> str:
        return "Data({})".format(",".join(self.splits))
    
    def __getitem__(self, key):
        return self.split[key]

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



if __name__ == "__main__":
    path = "data/Albanian/"
    data = Data(path)
    print(data)
    # feats = data.features()
    # for pos in feats:
    #     print(pos)
    #     print(feats[pos])