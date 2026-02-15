import torch
from mt.vocab import build_vocab
from mt.dataset import encode_file
from mt.model import Seq2Seq

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Paths
SRC_TRAIN = "Data/train.my.bpe"
TGT_TRAIN = "Data/train.en"
SRC_TEST = "Data/test.my.bpe"
MODEL_PATH = "mt_model.pt"
OUT_PATH = "pred.txt"

# Load vocab
src_vocab = build_vocab(SRC_TRAIN)
tgt_vocab = build_vocab(TGT_TRAIN)
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

# Load test data
src_data = encode_file(SRC_TEST, src_vocab)

# Load model
model = Seq2Seq(len(src_vocab), len(tgt_vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

import torch
import torch.nn.functional as F


def create_src_mask(src, pad_idx=0):
    """Create mask: 1 for real tokens, 0 for padding"""
    mask = (src != pad_idx).long()
    return mask


def beam_decode(src, model, inv_vocab, beam_size=3, max_len=50):
    """
    Beam search with attention
    """
    # Prepare source
    src = src.unsqueeze(0).to(DEVICE)  # (1, src_len)
    src_mask = create_src_mask(src, pad_idx=0).to(DEVICE)  # (1, src_len)

    with torch.no_grad():
        # Encode - get all encoder outputs for attention
        src_emb = model.src_emb(src)
        encoder_outputs, (h, c) = model.encoder(src_emb)
        # encoder_outputs: (1, src_len, hid_dim)

    # Each hypothesis: (token_ids, log_prob, h, c)
    beams = [([tgt_vocab["<s>"]], 0.0, h, c)]

    finished = []

    for _ in range(max_len):
        new_beams = []

        for tokens, score, h, c in beams:
            last_token = tokens[-1]

            # Stop expanding finished sequences
            if last_token == tgt_vocab["</s>"]:
                finished.append((tokens, score))
                continue

            inp = torch.tensor([[last_token]], device=DEVICE)  # (1, 1)

            with torch.no_grad():
                # Use the new decode_step method with attention
                logits, h_new, c_new, attn_weights = model.decode_step(
                    inp, h, c, encoder_outputs, src_mask
                )

                log_probs = F.log_softmax(logits, dim=-1)

            topk = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                next_tok = topk.indices[0, i].item()
                next_score = score + topk.values[0, i].item()
                new_beams.append(
                    (tokens + [next_tok], next_score, h_new, c_new)
                )

        # Keep best K beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if not beams:
            break

    finished.extend([(t, s) for t, s, _, _ in beams])

    best_tokens = max(finished, key=lambda x: x[1])[0]

    # Convert ids → tokens, remove <s> and </s>
    words = [
        inv_vocab.get(i, "<unk>")
        for i in best_tokens
        if i not in (tgt_vocab["<s>"], tgt_vocab["</s>"])
    ]

    return words


# Run inference
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for i, src in enumerate(src_data):
        tokens = beam_decode(
            src,
            model,
            inv_tgt_vocab,
            beam_size=3
        )
        f.write(" ".join(tokens) + "\n")

        if i % 1000 == 0:
            print(f"Translated {i}/{len(src_data)} sentences")

print("Inference complete → saved to pred.txt")