import torch

def encode_file(path, vocab, add_special=False):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if add_special:
                tokens = ["<s>"] + tokens + ["</s>"]
            data.append(torch.tensor(
                [vocab.get(w, 1) for w in tokens],
                dtype=torch.long
            ))
    return data

