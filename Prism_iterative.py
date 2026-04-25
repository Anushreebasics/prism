"""
==============================================================
 Prism Iterative (iPrism) — Multi-Round Self-Improvement
==============================================================

iPrism extends the core Prism framework into a self-refinement
loop. After each round of sampling + fine-tuning, the updated
expert model becomes the new expert for the next round. Because
the improved model generates better reasoning chains, the KLD
signal shifts toward progressively harder steps — enabling
compounding gains across rounds.

Round structure:
  1. Sample contrastive steps using expert_t vs. amateur
  2. Fine-tune expert_t → expert_{t+1} with LoRA + KL loss
  3. Merge LoRA adapter into base weights
  4. expert_{t+1} becomes new expert for round t+1

Usage:
------
  python Prism_iterative.py --rounds 3

Config:
-------
  Edit the CONFIG block below before running.

==============================================================
"""

import os
import json
import shutil
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# =============================================================
# CONFIG — edit before running
# =============================================================

base_expert_path   = "<path_to_base_expert>"      # e.g., "./Qwen2.5-Math-1.5B"
amateur_model_path = "<path_to_amateur>"           # e.g., "./Qwen2.5-0.5B"
input_data_path    = "<path_to_gsm8k_train_jsonl>" # e.g., "./gsm8k_train.jsonl"
work_dir           = "./iprism_rounds"             # directory for per-round artifacts

device     = "cuda"
torch_dtype = torch.bfloat16

# Sampling config
max_questions  = 1000
max_new_tokens = 128
alpha          = 0.2   # plausibility threshold
beta           = 0.4   # KLD selection threshold
batch_size     = 64

# Fine-tuning config
lora_rank    = 8
lora_alpha   = 16
ft_batch     = 8
ft_grad_acc  = 2
ft_lr        = 5e-5
ft_max_steps = 1000

# =============================================================
# Global precision optimization
# =============================================================
torch.set_float32_matmul_precision("high")


# =============================================================
# Dataset
# =============================================================
class ContrastiveSoftLabelDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, model_vocab_size, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab_size = model_vocab_size
        self.max_length = max_length
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user",   "content": item["prompt_id"]},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_input = formatted + item["prefix"]
        encoding = self.tokenizer(
            full_input, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        labels = torch.zeros(self.vocab_size, dtype=torch.float)
        for tid, w in zip(item["token_ids"], item["weights"]):
            if tid < self.vocab_size:
                labels[tid] = w
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# =============================================================
# Trainer
# =============================================================
class SoftLabelKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).logits
        vocab_size = inputs["labels"].size(-1)
        logits = logits[:, -1, :vocab_size]
        log_probs  = F.log_softmax(logits, dim=-1)
        loss = F.kl_div(log_probs, inputs["labels"], reduction="batchmean")
        return loss


def collate_fn(tokenizer, batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True,
            padding_value=tokenizer.pad_token_id,
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# =============================================================
# Sampling (inline, mirrors Prism_sampling.py)
# =============================================================
def run_sampling(expert_path, amateur_model, tokenizer, prompts, output_path, round_idx):
    print(f"\n[iPrism Round {round_idx}] Loading expert from {expert_path} for sampling...")
    expert_model = AutoModelForCausalLM.from_pretrained(
        expert_path, torch_dtype=torch_dtype, device_map="auto"
    ).eval()

    sampled = []
    checkpoint_path = output_path.replace(".jsonl", "_checkpoint.jsonl")
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass

    for prompt_obj in tqdm(prompts, desc=f"  Sampling (round {round_idx})"):
        prompt_id = prompt_obj["id"]
        if prompt_id in processed_ids:
            continue
        question = prompt_obj["question"]
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user",   "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = expert_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1]:].tolist()
        expert_probs   = [F.softmax(score[0], dim=-1) for score in outputs.scores]

        input_ids_batch = []
        for step in range(1, len(expert_targets)):
            prefix_decoded = tokenizer.decode(expert_targets[:step])
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted += prefix_decoded
            ids = tokenizer(formatted, return_tensors="pt").input_ids[0]
            input_ids_batch.append(ids)

        amateur_probs_all = []
        with torch.no_grad():
            for i in range(0, len(input_ids_batch), batch_size):
                chunk  = input_ids_batch[i:i + batch_size]
                padded = torch.nn.utils.rnn.pad_sequence(
                    chunk, batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(device, non_blocking=True)
                mask   = (padded != tokenizer.pad_token_id).long()
                logits = amateur_model(padded, attention_mask=mask).logits
                for j, seq_len in enumerate([x.size(0) for x in chunk]):
                    probs = F.softmax(logits[j, seq_len - 1, :], dim=-1)
                    amateur_probs_all.append(probs)

        count = 0
        for step in range(1, len(expert_targets)):
            ep = expert_probs[step]
            ap = amateur_probs_all[step - 1]
            min_v = min(ep.size(0), ap.size(0))
            ep, ap = ep[:min_v], ap[:min_v]

            eps  = 1e-12
            kld  = F.kl_div((ap + eps).log(), ep + eps, reduction="sum").item()
            if kld < beta:
                continue

            thresh = alpha * torch.max(ep)
            mask_v = ep >= thresh
            if not mask_v.any():
                continue

            log_PE = torch.log(ep + eps)
            log_PA = torch.log(ap + eps)
            scores = (log_PE - log_PA)[mask_v]
            weights = torch.softmax(scores, dim=-1)
            token_ids = torch.arange(ep.size(0), device=ep.device)[mask_v].tolist()

            sampled.append({
                "prompt_id":    prompt_id,
                "step":         step,
                "prefix":       tokenizer.decode(expert_targets[:step]),
                "token_ids":    token_ids,
                "weights":      weights.tolist(),
                "kl_divergence": kld,
            })
            count += 1

        with open(checkpoint_path, "a") as f:
            json.dump({"prompt_id": prompt_id, "num_samples": count}, f)
            f.write("\n")

    with open(output_path, "w") as f:
        for ex in sampled:
            json.dump(ex, f)
            f.write("\n")

    del expert_model
    torch.cuda.empty_cache()
    print(f"  Saved {len(sampled)} samples to {output_path}")
    return output_path


# =============================================================
# Fine-tuning
# =============================================================
def run_finetuning(expert_path, samples_path, output_dir):
    print(f"\n  Fine-tuning expert from {expert_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(expert_path)
    vocab_size = tokenizer.vocab_size

    base_model = AutoModelForCausalLM.from_pretrained(
        expert_path, torch_dtype=torch_dtype, device_map="auto"
    )
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)

    dataset = ContrastiveSoftLabelDataset(samples_path, tokenizer, vocab_size)
    import functools
    collate = functools.partial(collate_fn, tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=ft_batch,
        gradient_accumulation_steps=ft_grad_acc,
        learning_rate=ft_lr,
        max_steps=ft_max_steps,
        logging_steps=50,
        save_steps=ft_max_steps,
        save_total_limit=1,
        bf16=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SoftLabelKLTrainer(
        model=model, args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  LoRA checkpoint saved to {output_dir}")

    # Merge into base weights
    merged_dir = output_dir + "_merged"
    print(f"  Merging LoRA adapter into base weights → {merged_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        expert_path, torch_dtype=torch_dtype, device_map="cpu"
    )
    peft_model = PeftModel.from_pretrained(base, output_dir)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged model saved to {merged_dir}")

    del model, base, peft_model, merged
    torch.cuda.empty_cache()
    return merged_dir


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="iPrism: Iterative Prism Self-Improvement")
    parser.add_argument("--rounds", type=int, default=3, help="Number of iPrism rounds")
    args = parser.parse_args()

    os.makedirs(work_dir, exist_ok=True)

    # Load prompts once
    with open(input_data_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
    prompts = prompts[:max_questions]

    # Load amateur once — stays fixed across all rounds
    print("Loading amateur model (fixed across all rounds)...")
    amateur_model = AutoModelForCausalLM.from_pretrained(
        amateur_model_path, torch_dtype=torch_dtype, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(base_expert_path)

    # Track round results
    log_path = os.path.join(work_dir, "iprism_log.jsonl")
    current_expert = base_expert_path

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  iPrism — Round {round_idx} / {args.rounds}")
        print(f"{'='*60}")

        round_dir    = os.path.join(work_dir, f"round_{round_idx}")
        samples_path = os.path.join(round_dir, "samples.jsonl")
        ft_dir       = os.path.join(round_dir, "finetuned")
        os.makedirs(round_dir, exist_ok=True)

        # Stage 1: Sample
        run_sampling(current_expert, amateur_model, tokenizer, prompts, samples_path, round_idx)

        # Stage 2: Fine-tune + merge
        merged_path = run_finetuning(current_expert, samples_path, ft_dir)

        # Update expert for next round
        current_expert = merged_path

        # Log round summary
        n_samples = sum(1 for _ in open(samples_path))
        log_entry = {
            "round": round_idx,
            "samples": n_samples,
            "expert_out": merged_path,
        }
        with open(log_path, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        print(f"\n  Round {round_idx} complete. {n_samples} samples used.")

    print(f"\n{'='*60}")
    print(f"  iPrism complete. Final expert: {current_expert}")
    print(f"  Round log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
