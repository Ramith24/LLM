#!/usr/bin/env python3
"""
Full training pipeline for Burmese-English NMT with all improvements:
1. Data normalization (Zawgyi -> Unicode)
2. LoRA fine-tuning on NLLB-200-distilled-600M (Phase 4 + 5)
3. Optional: Scale to NLLB-200-1.3B with 8-bit quantization
4. Ensemble checkpoint averaging
5. MBR decoding with COMET-Kiwi
"""

import subprocess
import sys
import os

def run(cmd, desc, output_check_path=None):
    if output_check_path and os.path.exists(output_check_path):
        print(f"\n{'='*60}")
        print(f"SKIPPING: {desc}")
        print(f"Reason: Found existing output at {output_check_path}")
        print(f"{'='*60}")
        return True
        
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        return False
    return True

def main():
    steps = [
        # Step 1: Normalize Zawgyi to Unicode
        ("python mt/normalize_zawgyi_v2.py", "Normalize Burmese text: Zawgyi -> Unicode", None), # Runs quickly, safe to rerun
        
        # Step 2: Filter OPUS-100
        ("python mt/filter_opus.py", "Download and filter OPUS-100 en-my dataset", "Data/filtered_opus"),
        
        # Step 3: Back-translation
        ("python mt/back_translation.py", "Generate synthetic data via back-translation", "Data/synthetic_bt"),
        
        # Step 4: Fine-tune with LoRA on 600M model (Phase 4 + 5 combined)
        # We switched back to the 600M model to ensure it finishes within your 8-hour limit!
        ("python mt/finetune_phase5.py", "Fine-tune NLLB-600M with LoRA on augmented data", "hf_mt_model_phase5/final"),
        
        # Step 5: Ensemble checkpoints
        ("python mt/ensemble_checkpoints.py", "Average top-5 checkpoints for ensemble", "hf_mt_model_phase5/ensemble"),
        
        # Step 6: MBR decoding (requires COMET-Kiwi)
        ("python mt/mbr_decode.py", "MBR decoding with COMET-Kiwi", "Data/test_mbr_decoded.txt"),
    ]
    
    print("Burmese-English NMT Full Pipeline (Crash-Resilient Mode)")
    print("This will run all training steps sequentially.")
    print("If it crashes, just rerun this script and it will pick up exactly where it left off!")
    print("Estimated time: ~4-6 hours on RTX 4060")
    
    for cmd, desc, output_check in steps:
        if not run(cmd, desc, output_check):
            print(f"\nPipeline stopped at: {desc}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("Final model: hf_mt_model_phase5/final")
    print("Ensemble model: hf_mt_model_phase5/ensemble")
    print("="*60)

if __name__ == "__main__":
    main()