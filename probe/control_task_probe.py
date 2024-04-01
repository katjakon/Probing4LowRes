import random

from .probe_classifiers import ClassifierProbe

class ControlTaskProbe(ClassifierProbe):

    def __init__(self, data, train_repr, test_repr, clf_type, property="upos") -> None:
        super().__init__(data, train_repr, test_repr, clf_type, property)
        self.control_task_mapping = dict()
        self.n_classes = self.get_n_classes()
    
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
                self.control_task_mapping.setdefault(
                    word.form,
                    random.randrange(self.n_classes)
                )
                control_task_label = self.control_task_mapping[word.form]
                features.append(bert_repr[i])
                labels.append(control_task_label)
        return features, labels
    
    def get_n_classes(self):
        classes = set()
        for sent in self.data.train():
            for word in sent:
                if self.property == "upos":
                    ling_feat = word.upos
                else:
                    feats = word.feats
                    if self.property not in feats:
                        continue
                    ling_feat = feats[self.property]
                classes.add(ling_feat)
        return len(classes)
                