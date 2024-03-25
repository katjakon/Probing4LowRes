from transformers import BertTokenizer, BertModel

from utils.read_conllu import Data
from probe.probe import convert_to_ids, get_original_word_representations

PRETRAINED_MODEL = "google-bert/bert-base-multilingual-cased"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
model = BertModel.from_pretrained(PRETRAINED_MODEL)

data = Data("data/English")
train_data = data.train(verbose=True, sample_size=100)

train_instances = []
train_labels = []
for sent in train_data:
    words = [w.form for w in sent]
    pos = [w.upos for w in sent]
    mapping, ids = convert_to_ids(words, tokenizer=tokenizer)
    repr = get_original_word_representations(ids, model, mapping)

