import subprocess
import os
import sys

def run_experiment(config_name, use_w2v, teacher_forcing, emb_dim, hid_dim, num_layers, epochs):
    print(f"\n{'='*50}")
    print(f"Running Ablation Configuration: {config_name}")
    print(f"{'='*50}")
    
    # We would normally pass these as arguments to train.py, 
    # but since train.py is hardcoded, we generate a temporary training script
    # with the injected parameters.
    
    with open("mt/train.py", "r") as f:
        content = f.read()
        
    # Inject parameters
    content = content.replace("emb_dim=256", f"emb_dim={emb_dim}")
    content = content.replace("hid_dim=512", f"hid_dim={hid_dim}")
    content = content.replace("num_layers=2", f"num_layers={num_layers}")
    content = content.replace("EPOCHS = 15", f"EPOCHS = {epochs}")
    content = content.replace("teacher_forcing_ratio=0.5", f"teacher_forcing_ratio={teacher_forcing}")
    
    if not use_w2v:
        # Disable W2V by replacing the initialization block
        w2v_block = """print("Training custom Word2Vec embeddings on the fly...")"""
        content = content.replace(w2v_block, "pass # W2V Disabled\n\"\"\"")
        content = content.replace("print(\"Initialized encoder embeddings with FastText (.vec)\")", "\"\"\"\nprint(\"W2V Disabled\")")

    temp_script = f"mt/train_ablation_{config_name.replace(' ', '_').replace('+', 'plus')}.py"
    with open(temp_script, "w") as f:
        f.write(content)
        
    print(f"Running {temp_script}...")
    try:
        subprocess.run([sys.executable, temp_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        
    # Cleanup
    if os.path.exists(temp_script):
        os.remove(temp_script)

def main():
    experiments = [
        {
            "config_name": "Phase 1 Baseline",
            "use_w2v": False,
            "teacher_forcing": 1.0,  # 100% teacher forcing
            "emb_dim": 100,
            "hid_dim": 256,
            "num_layers": 1,
            "epochs": 5
        },
        {
            "config_name": "+ Word2Vec",
            "use_w2v": True,
            "teacher_forcing": 1.0,
            "emb_dim": 100,
            "hid_dim": 256,
            "num_layers": 1,
            "epochs": 5
        },
        {
            "config_name": "+ Scheduled Sampling",
            "use_w2v": True,
            "teacher_forcing": 0.5,
            "emb_dim": 100,
            "hid_dim": 256,
            "num_layers": 1,
            "epochs": 5
        },
        {
            "config_name": "+ Scaled Architecture (Phase 2 Model)",
            "use_w2v": True,
            "teacher_forcing": 0.5,
            "emb_dim": 256,
            "hid_dim": 512,
            "num_layers": 2,
            "epochs": 30
        }
    ]
    
    for exp in experiments:
        run_experiment(**exp)

if __name__ == "__main__":
    print("Warning: This ablation study will take significant time and GPU resources.")
    print("Ensure you monitor the system during execution.")
    main()
