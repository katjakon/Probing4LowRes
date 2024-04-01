from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

class ClassifierProbe:

    PROPERTIES = {
        "upos",
        "Number", 
        "Case",
        "Tense", 
        "Gender"
    }

    CLFS = {
        "SGD", 
        "MLP"
    }


    def __init__(self, data, train_repr, test_repr, clf_type, property="upos") -> None:
        self.data = data
        self.train_repr = train_repr
        self.test_repr = test_repr
        self.classes = None

        if property not in self.PROPERTIES:
            raise ValueError(f"{property} must be one of the following {self.PROPERTIES}")
        
        self.property = property

        if clf_type not in self.CLFS:
            raise ValueError(f"Invalid classifier type. Must be on of {self.CLFS}")
        
        if clf_type == "SGD":
            self.clf = SGDClassifier()
        if clf_type == "MLP":
            self.clf = MLPClassifier(hidden_layer_sizes=(2,))
    
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
            if self.property == "upos": #Extract POS tags
                bert_repr = repr[str(idx)]
                tags = [w.upos for w in sent][:bert_repr.shape[0]] # Truncate!
                features.extend(bert_repr)
                labels.extend(tags)
            else: # Extract given features of properties
                bert_repr = repr[str(idx)]
                for i in range(bert_repr.shape[0]):
                    feats = sent[i].feats
                    if self.property in feats:
                        labels.append(feats[self.property])
                        features.append(bert_repr[i])
        self.classes = set(labels)
        return features, labels

    def train(self):
        features, labels = self.get_features_and_labels("train")
        self.clf.fit(
            features, 
            labels
        )
    
    def test(self):
        features, labels = self.get_features_and_labels("test")
        return self.clf.predict(features), labels

    def evaluate(self):
        y_pred, y_true = self.test()
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        return {
            "Accuracy": acc,
            "Balanced Accuracy": balanced_acc
        }



    

