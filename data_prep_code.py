import json
import os
import argparse
from datasets import load_dataset

def prepare_code_data(output_path, max_samples=3000):
    """
    Downloads the 'm-a-p/CodeExercises' dataset or similar and formats it for Prism.
    Each record will have a 'question' and an 'id'.
    """
    print(f"Loading code dataset...")
    try:
        # Using a representative code dataset
        ds = load_dataset("m-a-p/CodeExercises", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load m-a-p/CodeExercises: {e}")
        print("Falling back to a standard code dataset (e.g., mbpp)...")
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train", streaming=True)

    samples = []
    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        
        # Standardizing format
        question = example.get("problem", example.get("text", ""))
        sample_id = f"code_{i}"
        
        samples.append({
            "id": sample_id,
            "question": question
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    
    print(f"Successfully saved {len(samples)} code samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Code Domain Data for Prism")
    parser.add_argument("--output", type=str, default="code_train.jsonl", help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=3000, help="Max samples to prepare")
    args = parser.parse_args()
    
    prepare_code_data(args.output, args.limit)
