from typing import List
class Data:

    def __init__(self, path: str) -> None:
        self.path = path # Path to conllu file.
        self.sents = self.read_conllu(self.path)
    
    def read_conllu(self, path: str):
        sents = []
        with open(path, encoding="utf-8") as file:
            sent = []
            for line in file:
                if line.strip():
                    if not line.startswith("#"):
                        sent.append(line.strip())
                else: # Empty line means a new sentence will start.
                    sents.append(sent)
                    sent = []
        return sents

class Sentence:

    def __init__(self, word_list: List[List[str]]) -> None:
        self.word_list = word_list
        self.read_words(word_list)

    def read_words(self, word_list):
        for word in word_list:
            w = Word(word.split("\t"))
            print(w)

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
        feats = feats.split("|")
        feat_dict = dict()
        for f in feats:
            feat_key, feat_value = f.split("=")
            feat_dict[feat_dict] = feat_value
        self._feats = feat_dict
        return self._feats

    def __repr__(self) -> str:
        return "{} {} {}".format(self.id, self.form, self.upos)



if __name__ == "__main__":
    path = "data/Russian/ru_syntagrus-ud-train-a.conllu"
    data = Data(path)
    for sent in data.sents[:1]:
        s = Sentence(sent)