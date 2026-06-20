import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mt.vocab import build_vocab

src_vocab = build_vocab("Data/train.my.bpe")
tgt_vocab = build_vocab("Data/train.en")

print(len(src_vocab), len(tgt_vocab))
