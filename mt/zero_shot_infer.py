import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading ZERO-SHOT model and tokenizer...")
# Local files only is False here because we may need to download the base model if not already cached.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(DEVICE)

print("Loading test data...")
src_sentences = []
with open("Data/test.my.bpe", "r", encoding="utf-8") as f:
    for line in f:
        # Note: NLLB has its own tokenizer, so feeding it BPE-tokenized text isn't ideal,
        # but since the baseline and phase 3 are evaluated on the same test set, we
        # strip the BPE tokens to pass raw text to the NLLB tokenizer.
        src_sentences.append(line.replace("@@ ", "").strip())

print(f"Translating {len(src_sentences)} sentences using ZERO-SHOT inference...")

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

print("Saving predictions to zero_shot_pred.txt...")
with open("zero_shot_pred.txt", "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")

print("Done! You can now grade this using sacrebleu.")
