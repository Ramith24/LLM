import sacrebleu

def compute_bleu(pred_file, ref_file):
    preds = open(pred_file, encoding="utf-8").read().splitlines()
    refs = [open(ref_file, encoding="utf-8").read().splitlines()]

    bleu = sacrebleu.corpus_bleu(preds, refs)
    print("BLEU:", bleu.score)
