import os
import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import evaluate

MODEL_CHECKPOINT = "facebook/nllb-200-distilled-600M"
SRC_LANG = "mya_Mymr"
TGT_LANG = "eng_Latn"

def main():
    print("=== Phase 5: Augmented Data Fine-Tuning ===")
    
    # 1. Load Local Data (The original Phase 4 data logic)
    print("Loading local datasets...")
    def load_and_clean_data(src_path, tgt_path):
        src_sentences, tgt_sentences = [], []
        with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_sentences.append(src_line.replace("@@ ", "").strip())
                tgt_sentences.append(tgt_line.strip())
        return {"translation": [{"my": s, "en": t} for s, t in zip(src_sentences, tgt_sentences)]}

    from datasets import Dataset
    train_dict = load_and_clean_data("Data/train.my.bpe", "Data/train.en")
    val_dict = load_and_clean_data("Data/val.my.bpe", "Data/val.en")
    local_train = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(val_dict)
    
    datasets_to_concat = [local_train]
    
    # 2. Load Filtered OPUS-100
    if os.path.exists("Data/filtered_opus"):
        print("Loading Filtered OPUS-100...")
        filtered_opus = load_from_disk("Data/filtered_opus")
        datasets_to_concat.append(filtered_opus)
    else:
        print("Warning: Filtered OPUS-100 not found. Run filter_opus.py first.")
        
    # 3. Load Synthetic Back-Translated Data
    if os.path.exists("Data/synthetic_bt"):
        print("Loading Synthetic Back-Translated Data...")
        synthetic_bt = load_from_disk("Data/synthetic_bt")
        datasets_to_concat.append(synthetic_bt)
    else:
        print("Warning: Synthetic BT data not found. Run back_translation.py first.")
        
    print("Merging datasets...")
    train_dataset = concatenate_datasets(datasets_to_concat).shuffle(seed=42)
    print(f"Final Augmented Training Set: {len(train_dataset)} pairs.")
    
    # Rest of the standard fine-tuning pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    
    def preprocess_function(examples):
        inputs = [ex["my"] for ex in examples["translation"]]
        targets = [ex["en"] for ex in examples["translation"]]
        return tokenizer(inputs, text_target=targets, max_length=64, truncation=True)
        
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["translation"])
    
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
        output_dir="hf_mt_model_phase5",
        eval_strategy="steps",
        eval_steps=2500,
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # Dropped to 2 because standard precision takes more memory
        gradient_accumulation_steps=8,  # Maintains effective batch size of 16
        num_train_epochs=5,             # Reduced from 15 to 5. 5 epochs is perfect for LoRA convergence.
        predict_with_generate=True,
        fp16=False,
        bf16=False, # Disabled half-precision completely to stop the CUBLAS crashes!
        save_strategy="steps",
        save_steps=2500,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=10,               # Print progress every 10 steps so it updates quickly!
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting Phase 5 Training...")
    
    last_checkpoint = None
    if os.path.isdir("hf_mt_model_phase5"):
        last_checkpoint = get_last_checkpoint("hf_mt_model_phase5")
        
    if last_checkpoint is not None:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model("hf_mt_model_phase5/final")
    print("Phase 5 Training Complete!")

if __name__ == "__main__":
    main()
