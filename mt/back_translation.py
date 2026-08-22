import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

def run_back_translation():
    print("Initializing Back-Translation Pipeline...")
    
    # Load a large monolingual English corpus
    # Using CC-100 English subset (large, diverse web text)
    print("Loading monolingual English data from CC-100...")
    try:
        mono_en_dataset = load_dataset("cc100", lang="en", split="train[:200000]")  # 200k sentences
    except Exception as e:
        print(f"Failed to load CC-100: {e}")
        print("Falling back to Wikipedia...")
        try:
            mono_en_dataset = load_dataset("wikipedia", "20220301.en", split="train[:50000]")
        except Exception as e2:
            print(f"Failed to load Wikipedia: {e2}")
            print("Using a dummy dataset for pipeline demonstration.")
            mono_en_dataset = [{"text": "This is a monolingual English sentence."}] * 1000
    
    # We will use the Phase 4 fine-tuned LoRA model for back-translation.
    # The direction is EN -> MY
    BASE_MODEL = "facebook/nllb-200-distilled-600M"
    LORA_PATH = "hf_mt_model/final" # Assuming phase 4 model is saved here
    
    SRC_LANG = "eng_Latn" # Source is now English
    TGT_LANG = "mya_Mymr" # Target is now Burmese
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Tokenizer and Model for En -> My translation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)
    
    # Try loading LoRA weights if available, otherwise fallback to base model
    if os.path.exists(LORA_PATH):
        from peft import PeftModel
        print(f"Loading Phase 4 LoRA weights from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)
    else:
        print(f"LoRA weights not found at {LORA_PATH}. Using base model.")
        model = base_model
        
    batch_size = 32
    synthetic_pairs = []
    
    # Extract sentences - filter for reasonable length
    sentences = []
    for item in mono_en_dataset:
        text = item["text"].strip()
        words = text.split()
        if 3 <= len(words) <= 50:  # Reasonable sentence length
            sentences.append(text)
    
    print(f"Translating {len(sentences)} English sentences into Burmese...")
    
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_length=64, 
                num_beams=4,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG)
            )
            
        decoded_my = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for en, my in zip(batch, decoded_my):
            # Filter: length ratio check to avoid hallucinated/empty translations
            if len(my.strip()) > 0 and 0.3 < len(en) / max(len(my), 1) < 3.5:
                synthetic_pairs.append({"translation": {"en": en, "my": my}})
                
    print(f"Generated {len(synthetic_pairs)} synthetic back-translated pairs.")
    
    synthetic_dataset = Dataset.from_list(synthetic_pairs)
    os.makedirs("Data/synthetic_bt", exist_ok=True)
    synthetic_dataset.save_to_disk("Data/synthetic_bt")
    print("Saved synthetic dataset to Data/synthetic_bt")

if __name__ == "__main__":
    run_back_translation()
