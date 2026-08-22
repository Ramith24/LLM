import subprocess
import os
import sys

def run_lora_experiment(rank, target_modules_str, run_name):
    print(f"\n{'='*50}")
    print(f"Running LoRA Configuration: {run_name} (Rank={rank})")
    print(f"{'='*50}")
    
    with open("mt/finetune.py", "r") as f:
        content = f.read()
        
    # Inject LoRA parameters
    content = content.replace("r=16,", f"r={rank},")
    content = content.replace('target_modules=["q_proj", "v_proj"]', f'target_modules={target_modules_str}')
    
    # Change output dir so they don't overwrite each other
    output_dir = f"hf_mt_model_{run_name.replace(' ', '_')}"
    content = content.replace('output_dir="hf_mt_model"', f'output_dir="{output_dir}"')
    
    # Run only 1 epoch for ablation speed
    content = content.replace("num_train_epochs=5,", "num_train_epochs=1,")

    temp_script = f"mt/finetune_ablation_{run_name.replace(' ', '_')}.py"
    with open(temp_script, "w") as f:
        f.write(content)
        
    print(f"Running {temp_script}...")
    try:
        subprocess.run([sys.executable, temp_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        
    # Cleanup script
    if os.path.exists(temp_script):
        os.remove(temp_script)

def main():
    experiments = [
        {
            "rank": 8,
            "target_modules_str": '["q_proj", "v_proj"]',
            "run_name": "rank_8_attn"
        },
        {
            "rank": 16,
            "target_modules_str": '["q_proj", "v_proj"]',
            "run_name": "rank_16_attn"
        },
        {
            "rank": 32,
            "target_modules_str": '["q_proj", "v_proj"]',
            "run_name": "rank_32_attn"
        },
        {
            "rank": 16,
            "target_modules_str": '["q_proj", "v_proj", "fc1", "fc2"]', # MLP layers for NLLB typically map to fc1 and fc2
            "run_name": "rank_16_mlp"
        }
    ]
    
    for exp in experiments:
        run_lora_experiment(**exp)

if __name__ == "__main__":
    print("Warning: This ablation study will take significant time and GPU resources.")
    print("Ensure you monitor the system during execution.")
    main()
