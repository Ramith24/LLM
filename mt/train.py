import torch
import torch.nn as nn
import torch.optim as optim
import torch

from mt.vocab import build_vocab
from mt.dataset import encode_file
from mt.model import Seq2Seq

# Load vocab
src_vocab = build_vocab("data/train.my.bpe")
tgt_vocab = build_vocab("data/train.en")

# Load data
src_data = encode_file("data/train.my.bpe", src_vocab)
tgt_data = encode_file("Data/train.en", tgt_vocab, add_special=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

model = Seq2Seq(len(src_vocab), len(tgt_vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

from gensim.models import KeyedVectors
import numpy as np

ft = KeyedVectors.load_word2vec_format(
    "Embeddings/fasttext_burmese.vec",
    binary=False
)

emb_dim = model.src_emb.embedding_dim
weights = np.random.randn(len(src_vocab), emb_dim)

for word, idx in src_vocab.items():
    if word in ft:
        weights[idx] = ft[word]

model.src_emb.weight.data.copy_(
    torch.tensor(weights, dtype=torch.float).to(DEVICE)
)

print("Initialized encoder embeddings with FastText (.vec)")

BATCH_SIZE = 32


def pad_batch(seqs, pad=0):
    max_len = max(len(s) for s in seqs)
    return torch.stack([
        torch.cat([s, torch.full((max_len - len(s),), pad)])
        for s in seqs
    ])


def create_src_mask(src, pad_idx=0):
    """
    Create mask for source sequences
    Args:
        src: (batch_size, src_len)
        pad_idx: padding token index
    Returns:
        mask: (batch_size, src_len) - 1 for real tokens, 0 for padding
    """
    mask = (src != pad_idx).long()
    return mask


EPOCHS = 5
for epoch in range(EPOCHS):
    total_loss = 0.0

    for i in range(0, len(src_data), BATCH_SIZE):
        src_batch = src_data[i:i + BATCH_SIZE]
        tgt_batch = tgt_data[i:i + BATCH_SIZE]

        src = pad_batch(src_batch).to(DEVICE)
        tgt = pad_batch(tgt_batch).to(DEVICE)

        # Create source mask for attention
        src_mask = create_src_mask(src, pad_idx=0).to(DEVICE)

        optimizer.zero_grad()

        # Pass src_mask to model
        output = model(src, tgt[:, :-1], src_mask)

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 5000 == 0:
            print(f"Processed {i}/{len(src_data)} sentences")

    print(f"Epoch {epoch + 1} loss: {total_loss:.2f}")

torch.save(model.state_dict(), "mt_model.pt")
print("Model saved as mt_model.pt")