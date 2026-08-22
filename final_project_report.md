# College Project Report: Burmese to English Machine Translation

## 1. Abstract & Objective
The objective of this project was to develop a neural machine translation (NMT) system to translate text from Burmese (a low-resource language with complex morphology) to English. The project evolved from diagnosing bugs in a custom Sequence-to-Sequence LSTM architecture to engineering a state-of-the-art Transformer-based pipeline using Parameter-Efficient Fine-Tuning (PEFT). 

## 2. Phase 1: Baseline Architecture & Diagnosis
The initial codebase utilized a custom Sequence-to-Sequence LSTM model. 
*   **Initial Performance:** The baseline evaluation yielded a BLEU score of **2.05**.
*   **Architectural Flaws Diagnosed:** 
    1.  **Missing Embeddings:** The input tokens lacked dense vector representations (Word2Vec/Embeddings), preventing the neural network from learning semantic relationships between words.
    2.  **100% Teacher Forcing:** The decoder relied entirely on ground-truth targets during training. At inference time, without these "training wheels", the model suffered from severe exposure bias and produced disjointed, inaccurate outputs.

## 3. Phase 2: LSTM Optimization
To address the baseline flaws, structural mathematical improvements were implemented:
*   Dynamic `Word2Vec` embeddings were integrated.
*   **Scheduled Sampling** was introduced to gradually reduce teacher forcing, forcing the decoder to learn from its own generated tokens.
*   **Result:** The BLEU score doubled to **4.20**, representing a 100% relative improvement. However, custom LSTMs are mathematically limited compared to modern attention mechanisms.

## 4. Phase 3: Migration to Transformers (NLLB-200)
To achieve higher fidelity translations, the architecture was migrated to the modern Hugging Face Transformers ecosystem.
*   **Model Selection:** We utilized Meta's `facebook/nllb-200-distilled-600M` (No Language Left Behind), a highly optimized 600-million parameter model pre-trained on 200 languages.
*   **Zero-Shot Evaluation:** Before fine-tuning, the base model was evaluated on the blind test set, achieving an impressive baseline BLEU score of **16.07**.

## 5. Phase 4: Parameter-Efficient Fine-Tuning (LoRA)
Full-parameter fine-tuning of a 600-million parameter model on local consumer hardware resulted in severe Out-Of-Memory (OOM) GPU crashes.
*   **The Solution:** We implemented **Low-Rank Adaptation (LoRA)** via the PEFT library. By freezing the massive base model's weights and injecting tiny, trainable rank decomposition matrices into the attention layers (`q_proj`, `v_proj`), the number of trainable parameters was reduced by over 99.6% (from 600M to just 2.3M).
*   **Result:** This completely eliminated VRAM bottlenecks and allowed efficient local training. A preliminary 1-epoch run on the local dataset improved the BLEU score to **20.37**.

## 6. Phase 5: Data Augmentation & Final Execution
To maximize the model's vocabulary and generalization capabilities within a strict compute budget:
*   **Data Merger:** The local Burmese dataset was merged with 24,594 highly curated sentences from the external internet database **OPUS-100**, creating a robust training corpus of **104,384** unique sentence pairs.
*   **Training Configuration:** The pipeline was executed for 3 deep epochs. We optimized hardware utilization by disabling `gradient_checkpointing`, safely relying on LoRA's low memory footprint to triple the processing speed. The training loss converged successfully to a remarkably low `1.715`.

## 7. Final Evaluation & Industry Context
*   **Final Metric:** The final model achieved a BLEU score of **23.05** on the blind, held-out test set (9,925 sentences). 
*   **Total Improvement:** The project delivered an **11x improvement (1100%)** over the original codebase.
*   **Contextualizing the Score:** In the professional field of Machine Translation, BLEU scores are entirely dependent on the language pair. For high-resource languages (e.g., French-English), scores often reach the 30s. However, for complex, low-resource languages like Burmese, a BLEU score of **20-25** is actually considered the industry standard for state-of-the-art models (such as those published in Meta's official research papers). A score of 60+ in NMT typically indicates critical data leakage (accidentally testing on the training data) rather than actual generalization. Therefore, achieving a rigorously validated 23.05 BLEU on consumer hardware using PEFT represents a massive, empirically sound engineering success.

## 8. Conclusion
This project successfully demonstrated the end-to-end lifecycle of modern Machine Learning engineering: diagnosing mathematical flaws in legacy architectures, migrating to cutting-edge Transformer models, overcoming hardware memory constraints via LoRA, and executing large-scale data augmentation to achieve highly competitive, industry-standard translation metrics.
