import os
from datasets import load_dataset, Dataset

def filter_opus100():
    print("Downloading OPUS-100 external dataset...")
    # OPUS-100 en-my
    opus_dataset = load_dataset("opus100", "en-my", split="train")
    
    print(f"Original OPUS-100 pairs: {len(opus_dataset)}")
    
    filtered_pairs = []
    seen = set()
    
    for item in opus_dataset:
        en_text = item["translation"]["en"].strip()
        my_text = item["translation"]["my"].strip()
        
        # 1. Remove empty sentences
        if not en_text or not my_text:
            continue
            
        # 2. Deduplication
        pair_hash = hash((en_text, my_text))
        if pair_hash in seen:
            continue
        seen.add(pair_hash)
        
        # 3. Length ratio filtering
        # A rough heuristic: translated sentences shouldn't be extremely disparate in length.
        # We compare character lengths.
        len_en = len(en_text)
        len_my = len(my_text)
        
        # Avoid division by zero
        if len_en < 2 or len_my < 2:
            continue
            
        ratio = len_en / len_my
        # If English is > 3x longer or < 0.3x shorter, it might be misaligned
        if ratio > 3.0 or ratio < 0.33:
            continue
            
        # 4. (Optional) Could add Zawgyi/Unicode normalization here using myanmar-tools
        
        filtered_pairs.append({"translation": {"en": en_text, "my": my_text}})
        
    print(f"Filtered OPUS-100 pairs: {len(filtered_pairs)}")
    print(f"Removed {len(opus_dataset) - len(filtered_pairs)} noisy/misaligned pairs.")
    
    filtered_dataset = Dataset.from_list(filtered_pairs)
    
    # Save the filtered dataset locally
    os.makedirs("Data/filtered_opus", exist_ok=True)
    filtered_dataset.save_to_disk("Data/filtered_opus")
    print("Saved filtered dataset to Data/filtered_opus")

if __name__ == "__main__":
    filter_opus100()
