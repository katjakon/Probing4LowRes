import numpy as np

from .probe_classifiers import ClassifierProbe


class RandomBaseline(ClassifierProbe):

    def __init__(self, data, train_repr, test_repr, clf_type, property="upos") -> None:
        super().__init__(data, train_repr, test_repr, clf_type, property)
        self.random_word_form_repr = dict()

    def get_features_and_labels(self, split):
        if split not in ("train", "test"):
            raise ValueError(f"{split} must be either 'train' or 'test'.")
        
        # Decide for train or test split
        if split == "train":
            sents = self.data.train()
            repr = self.train_repr
        else:
            sents = self.data.test()
            repr = self.test_repr
        features = []
        labels = []
        for idx, sent in enumerate(sents):
            bert_repr = repr[str(idx)]
            for i in range(bert_repr.shape[0]):
                word = sent[i]
                if self.property == "upos":
                    ling_feat = word.upos
                else:
                    feats = word.feats
                    if self.property not in feats:
                        continue
                    ling_feat = feats[self.property]
                self.random_word_form_repr.setdefault(
                    word.form, 
                    np.random.rand(*bert_repr[i].shape)
                )
                features.append(self.random_word_form_repr[word.form])
                labels.append(ling_feat)
        return features, labels
        