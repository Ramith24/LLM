import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = "facebook/nllb-200-distilled-600M"
LORA_PATH = "hf_mt_model/final"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG, local_files_only=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, use_safetensors=False, local_files_only=True)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(DEVICE)

print("Loading test data...")
src_sentences = []
with open("Data/test.my.bpe", "r", encoding="utf-8") as f:
    for line in f:
        src_sentences.append(line.replace("@@ ", "").strip())

print(f"Translating {len(src_sentences)} sentences...")

batch_size = 32
predictions = []

# Using tqdm for a progress bar
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

print("Saving predictions to hf_pred.txt...")
with open("hf_pred.txt", "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")

print("Done! You can now grade this using sacrebleu.")
