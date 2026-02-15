from mt.vocab import build_vocab

src_vocab = build_vocab("data/train.my.bpe")
tgt_vocab = build_vocab("data/train.en")

print(len(src_vocab), len(tgt_vocab))
