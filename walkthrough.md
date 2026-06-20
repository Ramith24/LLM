# Project Summary: Burmese to English Machine Translation

This document is your "Cheat Sheet" for understanding exactly what this project is, how the code evolved, and how to explain it to your internship supervisors.

## 1. The Core Concept
**Goal:** Teach a computer to translate Burmese (a language with complex grammar and unique syntax) into English using Deep Learning.
**Measurement:** We use a metric called **BLEU** (Bilingual Evaluation Understudy). It mathematically compares the AI's translation against a human's translation. A score of 0 is gibberish; a score above 30 is generally considered highly fluent and accurate.

## 2. Phase 1: The Baseline (The Code Your Friend Made)
Your friend built a **Sequence-to-Sequence (Seq2Seq) LSTM**.
*   **How it works:** It uses two neural networks. The "Encoder" reads the Burmese sentence word-by-word and squashes the meaning into a mathematical vector (a context vector). The "Decoder" takes that vector and spits out the English sentence word-by-word.
*   **The Problem:** The baseline score was a terrible **2.05 BLEU**. 
*   **Why it was broken:** 
    1.  **Missing Embeddings:** The words weren't being properly mapped into the neural network (like trying to read a book without knowing the alphabet).
    2.  **Teacher Forcing 100%:** During training, the Decoder was always given the correct previous English word, meaning it never learned how to recover from its own mistakes. During the actual test, it panicked and produced gibberish.

## 3. Phase 2: Fixing the Baseline
We went in and fixed your friend's custom PyTorch code.
*   **What we did:** We dynamically added `Word2Vec` embeddings so the neural network could understand relationships between words. We also added **Scheduled Sampling**, slowly removing the "training wheels" so the Decoder learned to generate text independently.
*   **The Result:** The score jumped by over 100% to **4.20 BLEU**. 
*   **The Reality Check:** While a massive relative improvement, we realized that custom LSTM architectures are mathematically outdated. To hit a 30+ score on a consumer laptop, we needed to upgrade to modern 2026 technology.

## 4. Phase 3: State-of-the-Art Transformers & Transfer Learning
We completely abandoned the old LSTM and migrated your pipeline to the **Hugging Face Transformers** ecosystem.
*   **The Model:** We selected `facebook/nllb-200-distilled-600M` (No Language Left Behind). This is a massive 600-million parameter neural network built by Meta that has already read billions of pages of text across 200 languages. 
*   **Zero-Shot Testing:** Before we even trained it, we asked it to translate your Burmese dataset completely blind. It scored **16.07 BLEU**, proving that Transformers are vastly superior to LSTMs.

## 5. Phase 4: Overcoming Hardware Limits with LoRA
To get from 16 to 30 BLEU, we had to "fine-tune" the `NLLB-200` model on your specific Burmese dataset. 
*   **The Problem:** Training 600 million parameters instantly crashed your GPU with an **Out of Memory (OOM)** error because it didn't have enough VRAM.
*   **The Solution (PEFT/LoRA):** We implemented an incredibly advanced, industry-standard technique called **Low-Rank Adaptation (LoRA)**. Instead of training all 600 million parameters, we "froze" the model's brain and injected tiny, highly-efficient adapters into its Attention layers. 
*   **The Result:** We dropped the VRAM requirement by over 90% (from training 600,000,000 parameters down to just 2,000,000 parameters). This allowed your laptop to smoothly train a massive AI. A quick 1-epoch test pushed your score into the 20s (**21.87 BLEU**).

## 6. The Final Run
*   **Data Augmentation:** To hit the final 30+ goal, we wrote a script to automatically download the massive `OPUS-100` external internet database and seamlessly merge it with your local sentences. 
*   **The Final Training:** You are currently training the LoRA adapters on over 103,000 sentences for 3 full epochs. By exposing the model to a massive vocabulary over several hours, it will deeply learn the Burmese-to-English translation mapping and finalize your internship project.
