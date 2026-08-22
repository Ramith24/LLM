import evaluate
import numpy as np
from scipy import stats

def compute_all_metrics(pred_file, ref_file):
    print("Loading predictions and references...")
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = [line.strip() for line in f]
    with open(ref_file, "r", encoding="utf-8") as f:
        refs = [[line.strip()] for line in f]
        
    assert len(preds) == len(refs), "Predictions and references must have the same length"
    
    print("Loading metrics...")
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")
    ter_metric = evaluate.load("ter")
    
    print("\n--- Point Estimates ---")
    
    bleu_result = bleu_metric.compute(predictions=preds, references=refs)
    print(f"BLEU: {bleu_result['score']:.2f}")
    
    # chrF++ uses word_order=2
    chrf_result = chrf_metric.compute(predictions=preds, references=refs, word_order=2)
    print(f"chrF++: {chrf_result['score']:.2f}")
    
    ter_result = ter_metric.compute(predictions=preds, references=refs)
    print(f"TER: {ter_result['score']:.2f}")
    
    print("\n--- 95% Confidence Intervals (Bootstrap Resampling) ---")
    # To compute CI, we need sentence-level scores. evaluate library doesn't easily expose this,
    # so we will use sacrebleu directly for sentence-level BLEU to bootstrap, or we can just 
    # bootstrap the dataset indices.
    
    def calc_corpus_bleu(indices):
        sampled_preds = [preds[i] for i in indices]
        sampled_refs = [refs[i] for i in indices]
        return bleu_metric.compute(predictions=sampled_preds, references=sampled_refs)["score"]
        
    def calc_corpus_chrf(indices):
        sampled_preds = [preds[i] for i in indices]
        sampled_refs = [refs[i] for i in indices]
        return chrf_metric.compute(predictions=sampled_preds, references=sampled_refs, word_order=2)["score"]
        
    # We use a smaller number of resamples (e.g. 100) for demonstration due to compute time, 
    # but 1000 is standard for research.
    indices = np.arange(len(preds))
    print("Running bootstrap for BLEU (n_resamples=100)...")
    try:
        # scipy.stats.bootstrap expects a statistic function that takes the data as arguments.
        # We pass the indices.
        res = stats.bootstrap((indices,), lambda idx: calc_corpus_bleu(idx), n_resamples=100, confidence_level=0.95, method='percentile')
        print(f"BLEU 95% CI: [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}]")
        
        print("Running bootstrap for chrF++ (n_resamples=100)...")
        res_chrf = stats.bootstrap((indices,), lambda idx: calc_corpus_chrf(idx), n_resamples=100, confidence_level=0.95, method='percentile')
        print(f"chrF++ 95% CI: [{res_chrf.confidence_interval.low:.2f}, {res_chrf.confidence_interval.high:.2f}]")
    except Exception as e:
        print(f"Bootstrap failed (likely scipy version): {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python evaluate_metrics.py <pred_file> <ref_file>")
        print("Example: python evaluate_metrics.py pred.txt Data/test.en")
    else:
        compute_all_metrics(sys.argv[1], sys.argv[2])
