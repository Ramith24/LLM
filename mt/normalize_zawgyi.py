#!/usr/bin/env python3
"""
Normalize Zawgyi-encoded Burmese text to Unicode.
Run this once to preprocess the data before training.
"""
import os
import re

# Zawgyi to Unicode character mapping
# Key mappings for common characters that differ
ZAWGYI_TO_UNICODE = {
    # Vowel signs - Zawgyi places them differently
    '\u1031': '\u102D',  # E vowel (Zawgyi: before consonant, Unicode: after)
    '\u1032': '\u102E',  # E vowel tall
    '\u1036': '\u1036',  # I vowel (same)
    '\u1037': '\u1037',  # I vowel tall
    '\u1038': '\u1038',  # U vowel
    '\u1039': '\u1039',  # U vowel tall
    '\u103A': '\u103A',  # E vowel (alternate)
    '\u103B': '\u103B',  # O vowel
    '\u103C': '\u103C',  # O vowel tall
    '\u103D': '\u103D',  # AU vowel
    '\u103E': '\u103E',  # AU vowel tall
    # Virama / killer
    '\u1039': '\u1039',  # Virama
    # Kinzi
    '\u1004\u103A': '\u1004\u103A',  # Kinzi
}

# Better approach: use myanmar-tools if available
def normalize_zawgyi_line(line):
    """Convert a line of Zawgyi text to Unicode"""
    # This is a simplified heuristic - for production use myanmar-tools library
    # Remove BPE markers first
    line = line.replace("@@ ", "").strip()
    
    # The main issue: in Zawgyi, the E vowel (U+1031) appears BEFORE the consonant
    # In Unicode, it appears AFTER the consonant
    # We need to reorder: consonant + U+1031 -> U+1031 + consonant
    
    # Myanmar consonant range: U+1000 to U+102A
    consonants = '[\u1000-\u102A]'
    e_vowel_zw = '\u1031'  # Zawgyi E vowel position
    
    # Pattern: E vowel followed by consonant (Zawgyi order)
    # Should become: consonant + E vowel (Unicode order)
    pattern = re.compile(f'({e_vowel_zw})({consonants})')
    
    def reorder(match):
        return match.group(2) + match.group(1)
    
    line = pattern.sub(reorder, line)
    
    return line

def preprocess_file(input_path, output_path):
    """Convert a file from Zawgyi to Unicode"""
    print(f"Processing {input_path} -> {output_path}")
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                normalized = normalize_zawgyi_line(line)
                f_out.write(normalized + '\n')
                if i % 10000 == 0 and i > 0:
                    print(f"  Processed {i} lines...")

def main():
    # Backup original files first
    import shutil
    for split in ['train', 'val', 'test']:
        src = f"Data/{split}.my.bpe"
        if os.path.exists(src):
            backup = f"Data/{split}.my.bpe.zawgyi_backup"
            if not os.path.exists(backup):
                shutil.copy2(src, backup)
                print(f"Backed up {src} to {backup}")
    
    # Process files
    for split in ['train', 'val', 'test']:
        src = f"Data/{split}.my.bpe"
        if os.path.exists(src):
            # Overwrite in place after backup
            temp = f"Data/{split}.my.bpe.unicode"
            preprocess_file(src, temp)
            shutil.move(temp, src)
            print(f"Normalized {src}")

if __name__ == "__main__":
    main()