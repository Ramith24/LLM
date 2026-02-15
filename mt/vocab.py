from collections import Counter

def build_vocab(path, min_freq=2):
    counter = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            counter.update(line.strip().split())

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3
    }

    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)

    return vocab
