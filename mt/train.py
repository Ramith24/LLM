import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim

from mt.vocab import build_vocab
from mt.dataset import encode_file
from mt.model import Seq2Seq

# Load vocab
src_vocab = build_vocab("Data/train.my.bpe")
tgt_vocab = build_vocab("Data/train.en")

# Load data
src_data = encode_file("Data/train.my.bpe", src_vocab)
tgt_data = encode_file("Data/train.en", tgt_vocab, add_special=True)
val_src_data = encode_file("Data/val.my.bpe", src_vocab)
val_tgt_data = encode_file("Data/val.en", tgt_vocab, add_special=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

model = Seq2Seq(len(src_vocab), len(tgt_vocab), emb_dim=256, hid_dim=512, num_layers=2, dropout=0.3).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

from gensim.models import Word2Vec
import numpy as np

emb_dim = model.src_emb.embedding_dim
weights = np.random.randn(len(src_vocab), emb_dim)

print("Training custom Word2Vec embeddings on the fly...")
sentences = []
with open("Data/train.my.bpe", encoding="utf-8") as f:
    for line in f:
        sentences.append(line.strip().split())

w2v_model = Word2Vec(sentences, vector_size=emb_dim, min_count=1, workers=4)

print("Loaded custom Word2Vec embeddings.")
for word, idx in src_vocab.items():
    if word in w2v_model.wv:
        weights[idx] = w2v_model.wv[word]

model.src_emb.weight.data.copy_(
    torch.tensor(weights, dtype=torch.float).to(DEVICE)
)

print("Initialized encoder embeddings with FastText (.vec)")

BATCH_SIZE = 128


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


import csv
with open("training_metrics.csv", "w") as f:
    f.write("epoch,train_loss,val_loss\n")

scaler = torch.amp.GradScaler('cuda')
EPOCHS = 15
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

        with torch.amp.autocast('cuda'):
            # Pass src_mask and teacher forcing to model
            # We can anneal teacher forcing, but let's keep it constant at 0.5 for now
            output = model(src, tgt[:, :-1], src_mask, teacher_forcing_ratio=0.5)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if i % 5000 == 0:
            print(f"Processed {i}/{len(src_data)} sentences")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(val_src_data), BATCH_SIZE):
            val_src_batch = val_src_data[i:i + BATCH_SIZE]
            val_tgt_batch = val_tgt_data[i:i + BATCH_SIZE]
            
            src = pad_batch(val_src_batch).to(DEVICE)
            tgt = pad_batch(val_tgt_batch).to(DEVICE)
            src_mask = create_src_mask(src, pad_idx=0).to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                output = model(src, tgt[:, :-1], src_mask, teacher_forcing_ratio=0.0) # No teacher forcing during val
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )
            val_loss += loss.item()
            
    avg_train_loss = total_loss / (len(src_data) / BATCH_SIZE)
    avg_val_loss = val_loss / (len(val_src_data) / BATCH_SIZE)
    print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save metrics to CSV
    with open("training_metrics.csv", "a") as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")

    model.train() # Set back to train mode
    torch.save(model.state_dict(), f"mt_model_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), "mt_model.pt")

print("Training complete. Model saved as mt_model.pt")