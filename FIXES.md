# Project Issues and Fixes

This document details the issues found during the initial evaluation of the codebase, preventing it from running successfully out of the box.

## 1. Missing Module Path (`ModuleNotFoundError`)
- **Location:** `mt/train.py`
- **Issue:** Running `python mt/train.py` from the root directory fails because Python cannot find the `mt` module. Python does not automatically add the root directory to its `PYTHONPATH`.
- **Fix Needed:** Either instruct the user to run the script as a module (`python -m mt.train`) or append the root directory to the system path at the top of the scripts.

## 2. Incorrect Embedding File Format (`FileNotFoundError`)
- **Location:** `mt/train.py` (Line 29)
- **Issue:** The training script tries to load FastText embeddings using `KeyedVectors.load_word2vec_format("Embeddings/fasttext_burmese.vec", binary=False)`. However, the provided file in the `Embeddings` directory is `fasttext_burmese.model`. The `.vec` file is missing, and the `load_word2vec_format` function cannot load the `.model` binary.
- **Fix Needed:** Change the embedding loading logic to use the proper format: `KeyedVectors.load("Embeddings/fasttext_burmese.model")` (or similar depending on how the FastText model was saved).

## 3. Inconsistent Path Casing
- **Location:** `mt/train.py` (Lines 11-16), `mt/infer.py` (Line 10), `test.py` (Line 3)
- **Issue:** The codebase uses inconsistent casing for the Data directory, alternating between `"data/..."` and `"Data/..."` (e.g., `data/train.my.bpe` vs `Data/train.en`). While this might work on Windows, it will cause immediate crashes on case-sensitive systems (Linux/macOS).
- **Fix Needed:** Standardize all file paths to use the correct casing `"Data/..."`.

## 4. Model Parameter Instantiation Mismatch
- **Location:** `mt/train.py` (Line 21) & `mt/model.py` (Line 53)
- **Issue:** In `train.py`, the model is instantiated as `Seq2Seq(len(src_vocab), len(tgt_vocab))`. However, the `Seq2Seq.__init__` in `model.py` is defined as `__init__(self, src_vocab, tgt_vocab, emb_dim=300, hid_dim=256)`. While it works due to positional arguments, it leaves `emb_dim` at the default 300 instead of matching the project documentation (which states 100).
- **Fix Needed:** Explicitly pass the correct hyperparameters when instantiating the `Seq2Seq` model in `train.py` to match the project's intended architecture.

## 5. Missing Dependency for Evaluation (`sacrebleu`)
- **Location:** `eval/bleu.py` (Line 1) & `README.md`
- **Issue:** The evaluation script imports the `sacrebleu` library (`import sacrebleu`), but this package is not listed in the installation instructions in the README. Running `python run_bleu.py` will fail with an `ImportError`.
- **Fix Needed:** Add `sacrebleu` to the pip installation instructions in the README.

## 6. Hardcoded Decoder State Shape
- **Location:** `mt/model.py` (Line 144)
- **Issue:** The `decode_step` method uses `h.squeeze(0)` during the attention calculation. This hardcodes the assumption that the LSTM has exactly 1 layer. If anyone tries to modify the model to include multiple layers (as the README suggests the architecture supports), the `squeeze(0)` operation will cause an immediate tensor shape mismatch during inference.
- **Fix Needed:** Refactor `Seq2Seq` to accept the `num_layers` and `dropout` arguments, and adjust the `decode_step` logic to correctly handle multi-layer hidden states (e.g., by selecting the hidden state from the top layer for attention).
