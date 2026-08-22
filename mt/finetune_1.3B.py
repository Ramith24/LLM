import os
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import evaluate
import warnings

# Suppress the noisy bitsandbytes quantization warnings that look like crashes
warnings.filterwarnings("ignore", message=".*inputs will be cast from.*")
warnings.filterwarnings("ignore", module="bitsandbytes.*")

# Use larger 1.3B model with 8-bit quantization
MODEL_CHECKPOINT = "facebook/nllb-200-1.3B"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"

def load_and_clean_data(src_path, tgt_path):
    src_sentences, tgt_sentences = [], []
    with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_sentences.append(src_line.replace("@@ ", "").strip())
            tgt_sentences.append(tgt_line.strip())
    return {"translation": [{"my": s, "en": t} for s, t in zip(src_sentences, tgt_sentences)]}

print("Loading datasets...")
train_dict = load_and_clean_data("Data/train.my.bpe", "Data/train.en")
val_dict = load_and_clean_data("Data/val.my.bpe", "Data/val.en")

from datasets import Dataset
train_dataset = Dataset.from_dict(train_dict)
val_dataset = Dataset.from_dict(val_dict)

# Load external datasets if available
datasets_to_concat = [train_dataset]
if os.path.exists("Data/filtered_opus"):
    print("Loading Filtered OPUS-100...")
    filtered_opus = load_from_disk("Data/filtered_opus")
    datasets_to_concat.append(filtered_opus)
if os.path.exists("Data/synthetic_bt"):
    print("Loading Synthetic Back-Translated Data...")
    synthetic_bt = load_from_disk("Data/synthetic_bt")
    datasets_to_concat.append(synthetic_bt)

train_dataset = concatenate_datasets(datasets_to_concat).shuffle(seed=42)
print(f"Final Training Set: {len(train_dataset)} pairs.")
print(f"Validation Set: {len(val_dataset)} pairs.")

print("Loading tokenizer and model with 8-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Applying LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

max_length = 64

def preprocess_function(examples):
    inputs = [ex["my"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[l.strip()] for l in decoded_labels]
    decoded_preds = [p.strip() for p in decoded_preds]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

args = Seq2SeqTrainingArguments(
    output_dir="hf_mt_model_1.3B",
    eval_strategy="steps",
    eval_steps=5650,
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Increased to speed up GPU usage
    gradient_accumulation_steps=2,
    num_train_epochs=3, # Reduced to 3 to finish in a reasonable time!
    predict_with_generate=True,
    fp16=False,  # 8-bit uses different precision
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    save_strategy="steps",
    save_steps=5650,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=10, # Changed from 500 so you see progress output quickly!
    dataloader_num_workers=0,
    gradient_checkpointing=True,  # Save memory
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

from transformers.trainer_utils import get_last_checkpoint
last_checkpoint = None
if os.path.isdir("hf_mt_model_1.3B"):
    last_checkpoint = get_last_checkpoint("hf_mt_model_1.3B")
    if last_checkpoint:
        print(f"Resuming from: {last_checkpoint}")

print("Starting Fine-Tuning on NLLB-1.3B...")
trainer.train(resume_from_checkpoint=last_checkpoint)

print("Saving final model...")
trainer.save_model("hf_mt_model_1.3B/final")
print("Done!")