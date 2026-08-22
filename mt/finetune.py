import os
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"

print("Loading dataset...")
def load_and_clean_data(src_path, tgt_path):
    src_sentences = []
    tgt_sentences = []
    
    with open(src_path, "r", encoding="utf-8") as f_src:
        for line in f_src:
            # Remove legacy BPE tokens (@@ )
            cleaned = line.replace("@@ ", "").strip()
            src_sentences.append(cleaned)
            
    with open(tgt_path, "r", encoding="utf-8") as f_tgt:
        for line in f_tgt:
            tgt_sentences.append(line.strip())
            
    # Ensure matching lengths
    assert len(src_sentences) == len(tgt_sentences)
    
    return {"translation": [{"my": src, "en": tgt} for src, tgt in zip(src_sentences, tgt_sentences)]}

# Load train and validation data
print("Loading local dataset...")
train_dict = load_and_clean_data("Data/train.my.bpe", "Data/train.en")
val_dict = load_and_clean_data("Data/val.my.bpe", "Data/val.en")

local_train = Dataset.from_dict(train_dict) # Keep all 79k
val_dataset = Dataset.from_dict(val_dict) # Keep all 10k

# We keep this script strictly for Phase 4 (LoRA fine-tuning on local dataset only).
# OPUS-100 augmentation is moved to a separate Phase 5 script.
train_dataset = local_train

print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang=SRC_LANG, tgt_lang=TGT_LANG, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT, use_safetensors=False, local_files_only=True)

print("Applying LoRA to the model...")
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
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np
import evaluate

metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # SacreBLEU requires targets to be in a list of lists
    decoded_labels = [[label.strip()] for label in decoded_labels]
    decoded_preds = [pred.strip() for pred in decoded_preds]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

args = Seq2SeqTrainingArguments(
    output_dir="hf_mt_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=5,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True,
    gradient_checkpointing=False,
    push_to_hub=False,
    logging_steps=500,
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
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

# Automatically detect the latest checkpoint to resume from if the script is interrupted
last_checkpoint = None
if os.path.isdir("hf_mt_model"):
    last_checkpoint = get_last_checkpoint("hf_mt_model")
    if last_checkpoint is not None:
        print(f"Resuming safely from checkpoint: {last_checkpoint}")

print("Starting Fine-Tuning...")
trainer.train(resume_from_checkpoint=last_checkpoint)

print("Saving final model...")
trainer.save_model("hf_mt_model/final")
print("Done!")
