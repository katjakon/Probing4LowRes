from sklearn.metrics import accuracy_score, balanced_accuracy_score

class MajorityBaseline:

    PROPERTIES = {
        "upos",
        "Number", 
        "Case",
        "Tense", 
        "Gender"
    }

    def __init__(self, data, property="upos") -> None:
        self.data = data

        if property not in self.PROPERTIES:
            raise ValueError(f"{property} must be one of the following {self.PROPERTIES}")
        
        self.property = property
        self.majority_votes = None
        self.most_common_overall = None

    
    def train(self):
        counts = dict()
        overall_counts = dict()
        for sent in self.data.train():
            for word in sent: 
                if self.property == "upos":
                    ling_feat = word.upos
                else:
                    feats = word.feats
                    if self.property not in feats:
                        continue
                    ling_feat = feats[self.property]
                counts.setdefault(word.form, dict())
                counts[word.form].setdefault(ling_feat, 0)
                overall_counts.setdefault(ling_feat, 0)
                counts[word.form][ling_feat] += 1
                overall_counts[ling_feat] += 1
        self.majority_votes = {
            word_form: max(counts[word_form], key=counts[word_form].get)
            for word_form in counts
        }
        self.most_common_overall = max(overall_counts, key=overall_counts.get)
    
    def test(self):
        y_pred = []
        y_true = []
        for sent in self.data.test():
            for word in sent:
                if self.property == "upos":
                    ling_feat = word.upos
                else:
                    feats = word.feats
                    if self.property not in feats:
                        continue
                    ling_feat = feats[self.property]
                pred = self.majority_votes.get(
                    word.form,
                    self.most_common_overall
                )
                y_true.append(ling_feat)
                y_pred.append(pred)
        return y_pred, y_true
    
    def evaluate(self):
        y_pred, y_true = self.test()
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        balanced_acc = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        return {
            "Accuracy": acc,
            "Balanced Accuracy": balanced_acc
        }

                






