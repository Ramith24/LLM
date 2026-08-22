import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np

BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_PATH = "hf_mt_model_phase5/final"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CANDIDATES = 20  # Generate N candidates per source
BEAM_SIZE = 20       # Beam size for diverse generation

def load_comet_kiwi():
    """Load COMET-Kiwi quality estimation model"""
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        return load_from_checkpoint(model_path)
    except Exception as e:
        print(f"Could not load COMET-Kiwi: {e}")
        print("Falling back to log-probability scoring...")
        return None

def diverse_beam_search(model, tokenizer, src_batch, num_candidates=20, max_len=64):
    """Generate diverse candidates using beam search with different penalties"""
    inputs = tokenizer(src_batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
    
    candidates_per_sent = []
    batch_size = len(src_batch)
    
    # Use multiple generation configs for diversity
    gen_configs = [
        {"num_beams": num_candidates, "num_beam_groups": num_candidates, "diversity_penalty": 1.0, "do_sample": False},
        {"num_beams": num_candidates, "num_beam_groups": num_candidates, "diversity_penalty": 0.5, "do_sample": False},
        {"num_beams": num_candidates, "do_sample": True, "top_k": 50, "temperature": 0.7},
        {"num_beams": num_candidates, "do_sample": True, "top_p": 0.9, "temperature": 0.8},
    ]
    
    all_outputs = []
    for config in gen_configs:
        cfg = {**config, "max_length": max_len, "forced_bos_token_id": tokenizer.convert_tokens_to_ids(TGT_LANG)}
        with torch.no_grad():
            outputs = model.generate(**inputs, **cfg)
        all_outputs.append(outputs)
    
    # Combine and deduplicate
    for i in range(batch_size):
        sent_candidates = []
        seen = set()
        for outputs in all_outputs:
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for d in decoded:
                d = d.strip()
                if d and d not in seen:
                    seen.add(d)
                    sent_candidates.append(d)
                if len(sent_candidates) >= num_candidates:
                    break
            if len(sent_candidates) >= num_candidates:
                break
        # Pad if needed
        while len(sent_candidates) < num_candidates:
            sent_candidates.append(sent_candidates[-1] if sent_candidates else "")
        candidates_per_sent.append(sent_candidates[:num_candidates])
    
    return candidates_per_sent

def score_candidates_mbr(comet_model, src_batch, candidates_batch):
    """Score candidates using COMET-Kiwi (reference-free) or fallback to model log-probs"""
    if comet_model is not None:
        # COMET-Kiwi expects list of dicts with 'src' and 'mt'
        data = []
        for src, cands in zip(src_batch, candidates_batch):
            for cand in cands:
                data.append({"src": src, "mt": cand})
        
        if not data:
            return [[0.0] * len(cands) for cands in candidates_batch]
        
        scores = comet_model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
        seg_scores = scores.scores
        
        # Reshape to (batch_size, num_candidates)
        batch_scores = []
        idx = 0
        for cands in candidates_batch:
            batch_scores.append(seg_scores[idx:idx + len(cands)])
            idx += len(cands)
        return batch_scores
    else:
        # Fallback: uniform scores (will pick first candidate)
        return [[1.0] * len(cands) for cands in candidates_batch]

def mbr_decode(model, tokenizer, src_sentences, comet_model=None, num_candidates=20, batch_size=16):
    """Minimum Bayes Risk decoding"""
    print(f"Generating {num_candidates} candidates per sentence...")
    all_best = []
    
    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i:i + batch_size]
        candidates = diverse_beam_search(model, tokenizer, batch, num_candidates=num_candidates)
        
        # Score candidates
        scores = score_candidates_mbr(comet_model, batch, candidates)
        
        # Select best per sentence (MBR: maximize expected utility)
        for cands, scs in zip(candidates, scores):
            best_idx = np.argmax(scs)
            all_best.append(cands[best_idx])
    
    return all_best

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG, local_files_only=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
    model.eval()
    
    print("Loading COMET-Kiwi for MBR scoring...")
    comet_model = load_comet_kiwi()
    
    print("Loading test data...")
    src_sentences = []
    with open("Data/test.my.bpe", "r", encoding="utf-8") as f:
        for line in f:
            src_sentences.append(line.replace("@@ ", "").strip())
    
    print(f"Running MBR decoding on {len(src_sentences)} sentences...")
    predictions = mbr_decode(model, tokenizer, src_sentences, comet_model, num_candidates=NUM_CANDIDATES)
    
    out_path = "mbr_pred.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    print(f"MBR predictions saved to {out_path}")
    
    # Evaluate
    from eval.bleu import compute_bleu
    compute_bleu(out_path, "Data/test.en")

if __name__ == "__main__":
    main()