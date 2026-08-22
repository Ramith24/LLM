# Appendix A: Reproducibility

To ensure the experiments documented in this manuscript can be fully reproduced, we provide the exact random seeds, data split methodologies, and preprocessing environments used.

## A.1 Random Seeds

All deep learning libraries were seeded prior to dataset shuffling, batching, and model initialization:
*   `torch.manual_seed(42)`
*   `numpy.random.seed(42)`
*   Hugging Face Datasets `.shuffle(seed=42)`

## A.2 Data Splits

The local Burmese-English dataset consists of approximately 99,000 parallel sentences. The exact, non-overlapping split methodology used is:
*   **Training Set (`train.my.bpe`, `train.en`):** 80% (79,200 pairs)
*   **Validation Set (`val.my.bpe`, `val.en`):** 10% (9,900 pairs)
*   **Test Set (`test.my.bpe`, `test.en`):** 10% (9,900 pairs)

*(Note: Exact line counts may slightly vary due to the removal of blank or severely malformed lines during the initial regex cleaning, yielding a final test set of 9,926 lines).*

## A.3 Preprocessing and Tokenization

### Byte Pair Encoding (BPE)
The local dataset was tokenized using Google's `SentencePiece`. 
*   **Vocabulary Size:** Target vocab size of 8,000 (`--vocab_size=8000`).
*   **Model Type:** BPE (`--model_type=bpe`).
*   **Normalization:** None (`--normalization_rule_name=identity`). *This was a methodological flaw analyzed in Phase 1, as it preserved mixed Zawgyi/Unicode encodings, artificially inflating the effective vocabulary.*

### OPUS-100 Filtering (Phase 5)
The external OPUS-100 dataset (`en-my` split) contains exactly 24,594 pairs. Before augmentation, it was filtered using the `filter_opus.py` script, which removes:
1.  Empty/null string translations.
2.  Exact duplicate pairs.
3.  Length-ratio outliers (Removing pairs where the character length ratio `en_len / my_len` is greater than 3.0 or less than 0.33, to eliminate gross misalignments).

## A.4 Software Environment

All models were trained on a generic Linux environment with CUDA 11.8 capability. Dependencies and exact versions are pinned in the provided `requirements.txt`. Key versions include:
*   Python 3.10+
*   PyTorch >= 2.0.0
*   Transformers >= 4.31.0
*   PEFT >= 0.4.0
*   SacreBLEU >= 2.3.1
