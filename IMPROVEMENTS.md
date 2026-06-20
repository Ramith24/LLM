# Project Improvements & Architecture Upgrades

This document outlines the root causes behind the initially low BLEU score (2.05) and details the architectural and training pipeline upgrades implemented to improve the model's translation performance.

## 1. Initial Limitations (Why the score was low)

The initial testing of the model yielded poor results due to four critical issues in the original setup:

- **Missing Pre-trained Embeddings:** The code relied on FastText `.npy` embedding files to understand the Burmese vocabulary. Because these files were missing from the repository, the model was forced to initialize with completely randomized embeddings, severely crippling its semantic understanding from the start.
- **Exposure Bias (100% Teacher Forcing):** The decoder was originally configured to receive the *correct* ground-truth target word at every single step during training. Because it was always spoon-fed the correct answer, it never learned how to recover from its own mistakes. During inference, when it had to rely on its own previous predictions, it quickly degraded into producing repetitive or random tokens.
- **Under-parameterized Architecture:** The original model was extremely small—configured with only 1 LSTM layer, a 100-dimensional embedding space, and a 256-dimensional hidden state. This lacked the necessary capacity (parameters) to capture the complex grammatical mappings between Burmese and English.
- **Insufficient Training Time:** The training loop was set to stop after just 5 epochs. Neural Machine Translation models typically require dozens of epochs to fully converge; stopping at 5 meant the model barely began to learn the dataset.

---

## 2. Implemented Upgrades (How we fixed it)

To address these weaknesses and significantly boost the BLEU score, the following upgrades were engineered and deployed:

- **On-the-Fly Custom Embeddings:**
  To compensate for the missing FastText files, a dynamic embedding generation script was integrated into `train.py`. The pipeline now trains a custom `Word2Vec` model directly on the `train.my.bpe` dataset right before the Seq2Seq training begins. This provides the model with strong, context-aware initial word representations.
  
- **Scheduled Sampling (Teacher Forcing Ratio):**
  We modified the forward pass in `model.py` to include a `teacher_forcing_ratio` of 50%. Now, during training, the model randomly decides (50% of the time) whether to use the correct ground-truth word or its own predicted word from the previous step. This forces the model to become self-reliant and directly mitigates exposure bias during inference.
  
- **Scaled Model Architecture:**
  We significantly expanded the model's capacity to handle complex translations:
  - Doubled the embedding dimensions from `100` to `256`.
  - Doubled the hidden state dimensions from `256` to `512`.
  - Added a second LSTM layer (`num_layers=2`) for deeper sequential processing.
  - Introduced a `30%` Dropout rate to prevent the newly enlarged network from overfitting on the training data.

- **Extended Training Duration:**
  The `EPOCHS` parameter was increased from `5` to `30`, providing the larger architecture with adequate time to properly converge and map the linguistic structures.
