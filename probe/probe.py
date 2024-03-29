
import torch

def get_original_word_representations(ids, model, mapping, layer_idx=-1):
    output = model(ids, output_hidden_states=True)
    output = torch.squeeze(output[layer_idx][0], dim=0)
    # We don't need representations of [CLS] and [SEP]
    output = output[1:-1]
    # Average representations of subwords for the mapped indices.
    aggr_repr = torch.zeros(size=(len(mapping), output.shape[-1]))
    for idx, (start, end) in enumerate(mapping):
        if end - start > 1:
            mean = torch.mean(output[start:end], dim=0)
            aggr_repr[idx] += mean
        else:
            aggr_repr[idx] += output[start]
    return aggr_repr

def convert_to_ids(word_list, tokenizer, max_len=512):
    tokenized = []
    mapping = []
    subwords_idx = 0
    for word in word_list:
        subwords = tokenizer.tokenize(word)
        length = len(subwords)
        new_idx = subwords_idx + length
        if new_idx + 2 > max_len:
            break
        tokenized.extend(subwords)
        mapping.append((subwords_idx, new_idx))
        subwords_idx = new_idx
    input_seq = ["[CLS]"] + tokenized + ["[SEP]"]
    ids = tokenizer.convert_tokens_to_ids(input_seq)
    return mapping, torch.unsqueeze(torch.IntTensor(ids), dim=0)