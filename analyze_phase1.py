import sys
import os

try:
    from myanmar import language
except ImportError:
    pass # we will use another way if myanmar-tools is not available

try:
    from markovzawgyi import markovzawgyi
except ImportError:
    pass

# We will just write a script to calculate vocab sizes and OOV rate first.
# Then we will write a script to check for zawgyi.
def analyze():
    # 1. Vocab size
    def build_vocab(filepath):
        vocab = set()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                vocab.update(line.strip().split())
        return vocab

    train_vocab = build_vocab("Data/train.my.bpe")
    test_vocab = build_vocab("Data/test.my.bpe")
    val_vocab = build_vocab("Data/val.my.bpe") if os.path.exists("Data/val.my.bpe") else set()

    print(f"Training Vocab Size: {len(train_vocab)}")
    print(f"Test Vocab Size: {len(test_vocab)}")
    
    # 2. OOV rate
    oov_tokens = test_vocab - train_vocab
    oov_rate = len(oov_tokens) / len(test_vocab) * 100 if len(test_vocab) > 0 else 0
    print(f"Test OOV Tokens: {len(oov_tokens)}")
    print(f"Test OOV Rate: {oov_rate:.2f}%")
    
    # 3. Encoding Distribution (Heuristic based on \u1031 placement)
    import re
    def detect_encoding(filepath):
        # Zawgyi types \u1031 (e vowel) before consonants. Unicode types it after.
        zawgyi_pattern = re.compile(r'\u1031[\u1000-\u102A]')
        unicode_pattern = re.compile(r'[\u1000-\u102A]\u1031')
        
        zawgyi_count = 0
        unicode_count = 0
        unknown_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove spaces and BPE tokens ' ' (\u2581) to reconstruct words
                text = line.replace(' ', '').replace(' ', '').strip()
                if not text: continue
                
                has_z = bool(zawgyi_pattern.search(text))
                has_u = bool(unicode_pattern.search(text))
                
                if has_z and not has_u:
                    zawgyi_count += 1
                elif has_u and not has_z:
                    unicode_count += 1
                elif has_z and has_u:
                    # Mixed in the same line
                    zawgyi_count += 0.5
                    unicode_count += 0.5
                else:
                    unknown_count += 1
                    
        return zawgyi_count, unicode_count, unknown_count

    print("\n--- Encoding Distribution ---")
    z, u, un = detect_encoding("Data/train.my.bpe")
    print(f"Train Data - Zawgyi: {z}, Unicode: {u}, Unknown: {un}")
    z, u, un = detect_encoding("Data/test.my.bpe")
    print(f"Test Data  - Zawgyi: {z}, Unicode: {u}, Unknown: {un}")

if __name__ == "__main__":
    analyze()
