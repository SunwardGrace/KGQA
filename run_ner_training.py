#!/usr/bin/env python3
"""
Quick script to train NER model on CMeEE dataset.

Usage:
    python run_ner_training.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def main():
    train_file = PROJECT_ROOT / "mydata" / "cmeee.json"
    output_dir = PROJECT_ROOT / "models" / "ner_cmeee"

    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "nlp.ner.train_cmeee",
        "--train_file", str(train_file),
        "--output_dir", str(output_dir),
        "--model_name", "bert-base-chinese",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "8",
        "--gradient_accumulation_steps", "2",
        "--max_length", "128",
        "--valid_ratio", "0.1",
        "--fp16",
        "--logging_steps", "100",
        "--overwrite_output_dir",
    ]

    print("=" * 60)
    print("Starting NER Training on CMeEE Dataset")
    print("=" * 60)
    print(f"Train file: {train_file}")
    print(f"Output dir: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print("\nTo use the trained model:")
        print('  from nlp.ner.extractor import NERExtractor')
        print(f'  extractor = NERExtractor(model_name="{output_dir}", device="cuda:0")')
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
