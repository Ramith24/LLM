# Burmese-to-English Machine Translation

A sequence-to-sequence neural machine translation model with attention mechanism for translating Burmese text to English.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Models & Embeddings](#models--embeddings)
- [File Descriptions](#file-descriptions)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a neural machine translation (NMT) system that translates Burmese text to English using:
- **Seq2Seq architecture** with LSTM encoder-decoder
- **Luong attention mechanism** for improved translation quality
- **FastText embeddings** for Burmese language representations
- **Byte Pair Encoding (BPE)** tokenization for both languages
- **Beam search decoding** for inference

## Features

✅ Attention-based encoder-decoder architecture
✅ FastText pre-trained embeddings for Burmese
✅ BPE tokenization for subword units
✅ Beam search decoding with configurable beam size
✅ BLEU score evaluation
✅ GPU support (CUDA)
✅ Modular code structure (dataset, model, training, inference)

## Project Structure

```
LLM/
├── Data/                           # Training and test data
│   ├── train.my.bpe              # Training Burmese (tokenized)
│   ├── train.en                  # Training English
│   ├── val.my.bpe                # Validation Burmese (tokenized)
│   ├── val.en                    # Validation English
│   ├── test.my.bpe               # Test Burmese (tokenized)
│   └── test.en                   # Test English (reference)
│
├── Embeddings/                     # Pre-trained embeddings
│   └── fasttext_burmese.model    # FastText Burmese embeddings
│
├── Tokenizer/                      # Tokenization models
│   └── spm_burmese.model         # SentencePiece Burmese tokenizer
│
├── eval/                           # Evaluation scripts
│   └── bleu.py                   # BLEU score computation
│
├── mt/                             # Main module
│   ├── __init__.py
│   ├── model.py                  # Seq2Seq model with attention
│   ├── train.py                  # Training script
│   ├── infer.py                  # Inference/prediction script
│   ├── dataset.py                # Data loading utilities
│   └── vocab.py                  # Vocabulary building
│
├── run_bleu.py                     # Evaluate predictions with BLEU score
├── test.py                         # Test vocabulary sizes
├── pred.txt                        # Predictions output file
└── README.md                       # This file
```

## Prerequisites

- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy
- Gensim
- SentencePiece (optional, for tokenization)

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd LLM
```

### 2. Create a virtual environment (recommended)
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy gensim sentencepiece sacrebleu
```

Or install with GPU support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy gensim sentencepiece sacrebleu
```

### 4. Verify installation
```bash
python test.py
# Should output vocabulary sizes
```

## Dataset

### Data Format
- **Source (Burmese)**: UTF-8 encoded, BPE tokenized (space-separated subword units)
- **Target (English)**: UTF-8 encoded, space-separated words

### Data Files
| File | Purpose | Contains |
|------|---------|----------|
| `train.my.bpe` | Training source | ~X,XXX sentences (Burmese, BPE) |
| `train.en` | Training target | ~X,XXX sentences (English) |
| `val.my.bpe` | Validation source | Burmese sentences (BPE) |
| `val.en` | Validation target | English sentences |
| `test.my.bpe` | Test source | Burmese sentences (BPE) |
| `test.en` | Test reference | English sentences (for evaluation) |

### Using Your Own Data
1. Prepare parallel text files (one sentence per line)
2. Tokenize Burmese text with BPE:
   ```bash
   spm_encode --model=Tokenizer/spm_burmese.model < your_burmese.txt > your_burmese.bpe
   ```
3. Place in `Data/` directory
4. Update file paths in training scripts

## Quick Start

### Training

1. **Start training:**
   ```bash
   python mt/train.py
   ```

2. **Training process:**
   - Loads vocabularies from training data
   - Initializes FastText embeddings for Burmese
   - Trains Seq2Seq model with cross-entropy loss
   - Saves checkpoint to `mt_model.pt`
   - Prints loss after each batch

3. **Key hyperparameters in `train.py`:**
   - `BATCH_SIZE`: 32 (adjust based on GPU memory)
   - `EPOCHS`: Modify `for epoch in range(num_epochs)`
   - `LEARNING_RATE`: Controlled by Adam optimizer

### Inference

1. **Generate predictions:**
   ```bash
   python mt/infer.py
   ```

2. **Output:**
   - Generates translations for test data
   - Saves to `pred.txt` (one translation per line)
   - Uses beam search with default beam_size=3

3. **Customize inference:**
   - Edit `beam_size` parameter in `beam_decode()` function
   - Adjust `max_len` for maximum translation length
   - Change input/output paths in script

### Evaluation

1. **Calculate BLEU score:**
   ```bash
   python run_bleu.py
   ```

2. **Output:**
   - Displays BLEU score comparing `pred.txt` with `Data/test.en`
   - Also shows BLEU breakdown (unigram, bigram, trigram, 4-gram)

## Configuration

### Model Architecture
Edit `mt/model.py` to modify:
```python
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 emb_dim=100, hidden_dim=256, num_layers=2, dropout=0.3):
```

### Training Parameters
Edit `mt/train.py`:
- `BATCH_SIZE`: Number of samples per batch
- `DEVICE`: "cuda" for GPU, "cpu" for CPU
- Add validation/checkpoint saving in the training loop

### Inference Parameters
Edit `mt/infer.py`:
- `beam_size`: Larger = better but slower (default: 3)
- `max_len`: Maximum translation length (default: 50)
- `MODEL_PATH`: Path to trained model checkpoint

## Models & Embeddings

### FastText Embeddings
- **File**: `Embeddings/fasttext_burmese.model`
- **Purpose**: Pre-trained word vectors for Burmese
- **Usage**: Automatically loaded in `train.py` to initialize encoder embeddings
- **Format**: Word2Vec format (.vec)

### Sentencepiece Tokenizer
- **File**: `Tokenizer/spm_burmese.model`
- **Purpose**: Byte pair encoding for Burmese text
- **Usage**: Tokenize new Burmese text before translation

### Trained Model
- **File**: `mt_model.pt` (generated after training)
- **Size**: Depends on vocabulary and hidden dimensions
- **Contains**: All model weights and parameters
- **Load**: `torch.load("mt_model.pt", map_location=DEVICE)`

## File Descriptions

### Core Module Files (`mt/`)

#### `model.py`
- **LuongAttention**: Multiplicative attention mechanism
  - Aligns encoder outputs with decoder hidden state
  - Returns attention weights and context vector
- **Seq2Seq**: Main model architecture
  - Encoder: LSTM with embeddings
  - Decoder: LSTM with attention
  - Attention computation and output projection

#### `train.py`
- Data loading and vocabulary building
- FastText embedding initialization
- Batch padding and processing
- Training loop with loss computation
- Outputs: `mt_model.pt`

#### `infer.py`
- Model loading and evaluation mode
- Beam search decoding algorithm
- Batch inference capabilities
- Predictions saved to `pred.txt`

#### `dataset.py`
- `encode_file()`: Convert text to token IDs using vocabulary
- Handles special tokens (`<s>`, `</s>`)
- Returns PyTorch tensors

#### `vocab.py`
- `build_vocab()`: Creates token-to-ID mappings from text files
- Handles unknown tokens (ID=1)
- Padding tokens (ID=0)

### Evaluation Files (`eval/`)

#### `bleu.py`
- Computes BLEU score
- Compares predictions with reference translations
- Returns sentence-level and corpus-level scores

### Utility Scripts

#### `test.py`
- Quick test to verify vocabulary loading
- Outputs source and target vocabulary sizes
- Use: `python test.py`

#### `run_bleu.py`
- Evaluates current predictions
- Compares `pred.txt` with `Data/test.en`
- Use: `python run_bleu.py`

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Data/train.my.bpe`
- **Solution**: Ensure Data directory exists and files are in correct location
- Check file paths in scripts (some use `data/`, some use `Data/`)
- Standardize directory paths

**Issue**: `CUDA out of memory`
- **Solution**: Reduce `BATCH_SIZE` in training script (try 16 or 8)
- Use CPU: `DEVICE = torch.device("cpu")`

**Issue**: Low BLEU scores
- **Solution**: 
  - Train for more epochs
  - Increase model hidden_dim
  - Add regularization (dropout)
  - Check data quality and alignment

**Issue**: `KeyError` when building vocabulary
- **Solution**: Ensure training data files exist and contain text
- Verify UTF-8 encoding of data files

**Issue**: Predictions are all same translations
- **Solution**: Model may not be converging
  - Train longer with lower learning rate
  - Check if training loss is decreasing
  - Verify data preprocessing

### Getting Help

1. Check error messages carefully - they usually indicate the problem
2. Verify all file paths are correct
3. Ensure all dependencies are installed: `pip list`
4. Test individual components:
   ```bash
   python test.py  # Check vocab loading
   python run_bleu.py  # Check evaluation
   ```
5. Print debug information in scripts to trace issues

## Next Steps

1. **Verify setup**: Run `python test.py`
2. **Train model**: Run `python mt/train.py`
3. **Generate predictions**: Run `python mt/infer.py`
4. **Evaluate results**: Run `python run_bleu.py`
5. **Experiment**: Adjust hyperparameters and retrain
6. **Deploy**: Use trained model for inference in production

## Additional Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- BLEU Score Paper: https://www.aclweb.org/anthology/P02-1040.pdf
- FastText: https://fasttext.cc/
- Byte Pair Encoding: https://arxiv.org/abs/1508.07909