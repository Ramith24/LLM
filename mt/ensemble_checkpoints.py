import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file, save_file
from tqdm import tqdm

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = "hf_mt_model_phase5"
OUTPUT_DIR = "hf_mt_model_phase5/ensemble"
TOP_K = 5

def get_checkpoint_scores():
    """Get validation BLEU scores from trainer_state.json files"""
    import json
    checkpoints = []
    for item in os.listdir(CHECKPOINT_DIR):
        if item.startswith("checkpoint-"):
            state_path = os.path.join(CHECKPOINT_DIR, item, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                # Get the latest eval_bleu
                best_bleu = 0
                for log in state.get("log_history", []):
                    if "eval_bleu" in log:
                        best_bleu = max(best_bleu, log["eval_bleu"])
                if best_bleu > 0:
                    checkpoints.append((item, best_bleu))
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints

def average_checkpoints(checkpoint_names):
    """Average LoRA adapter weights from multiple checkpoints"""
    print(f"Averaging {len(checkpoint_names)} checkpoints...")
    
    # Load first checkpoint to get structure
    first_path = os.path.join(CHECKPOINT_DIR, checkpoint_names[0], "adapter_model.safetensors")
    state_dict = load_file(first_path)
    
    # Initialize accumulator
    avg_state = {k: v.clone().float() for k, v in state_dict.items()}
    
    # Add remaining checkpoints
    for name in checkpoint_names[1:]:
        path = os.path.join(CHECKPOINT_DIR, name, "adapter_model.safetensors")
        sd = load_file(path)
        for k in avg_state:
            avg_state[k] += sd[k].float()
    
    # Average
    for k in avg_state:
        avg_state[k] /= len(checkpoint_names)
    
    # Convert back to original dtype
    avg_state = {k: v.to(state_dict[k].dtype) for k, v in avg_state.items()}
    
    # Save ensemble
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(avg_state, os.path.join(OUTPUT_DIR, "adapter_model.safetensors"))
    
    # Copy adapter_config.json
    import shutil
    shutil.copy(
        os.path.join(CHECKPOINT_DIR, checkpoint_names[0], "adapter_config.json"),
        os.path.join(OUTPUT_DIR, "adapter_config.json")
    )
    
    print(f"Ensemble saved to {OUTPUT_DIR}")
    return OUTPUT_DIR

def evaluate_ensemble(ensemble_path):
    """Run inference with ensemble model"""
    print("Evaluating ensemble model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG, local_files_only=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, use_safetensors=False, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, ensemble_path).to(DEVICE)
    model.eval()

    src_sentences = []
    with open("Data/test.my.bpe", "r", encoding="utf-8") as f:
        for line in f:
            src_sentences.append(line.replace("@@ ", "").strip())

    batch_size = 32
    predictions = []

    for i in tqdm(range(0, len(src_sentences), batch_size)):
        batch = src_sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=64, 
                num_beams=4,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG)
            )
            
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    out_path = os.path.join(OUTPUT_DIR, "ensemble_pred.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
    
    print(f"Predictions saved to {out_path}")
    return out_path

if __name__ == "__main__":
    checkpoints = get_checkpoint_scores()
    print(f"Found {len(checkpoints)} checkpoints with BLEU scores:")
    for name, score in checkpoints[:10]:
        print(f"  {name}: {score:.2f}")
    
    top_checkpoints = [name for name, _ in checkpoints[:TOP_K]]
    print(f"\nAveraging top {TOP_K}: {top_checkpoints}")
    
    ensemble_path = average_checkpoints(top_checkpoints)
    pred_file = evaluate_ensemble(ensemble_path)
    
    # Run BLEU evaluation
    from eval.bleu import compute_bleu
    print("\nEvaluating ensemble BLEU...")
    compute_bleu(pred_file, "Data/test.en")