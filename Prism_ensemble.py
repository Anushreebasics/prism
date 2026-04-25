"""
==============================================================
 Prism Ensemble — Multi-Amateur Contrastive Sampling
==============================================================

Extends Prism sampling by aggregating contrastive signals from
N amateur models. Step selection requires average KLD across
all amateurs to exceed beta. Soft labels are weighted averages
of each amateur's contrastive distribution, with weights
proportional to expertise gap.

Usage:
------
  python Prism_ensemble.py \
    --amateur_paths ./Qwen2.5-0.5B,./Llama-3.2-1B \
    --amateur_gaps 38.2,31.0 \
    --max_questions 1000

Output: same JSONL format as Prism_sampling.py.
==============================================================
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================
# CONFIG
# =============================================================
expert_model_path = "<path_to_expert_model>"
input_path        = "<path_to_gsm8k_train_jsonl>"
output_path       = "<path_to_output_jsonl>"
checkpoint_path   = "<path_to_checkpoint_jsonl>"

device      = "cuda"
torch_dtype = torch.bfloat16

max_new_tokens = 128
alpha          = 0.2
beta           = 0.4
batch_size     = 64

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amateur_paths", type=str, required=True,
                        help="Comma-separated list of amateur model paths")
    parser.add_argument("--amateur_gaps", type=str, default=None,
                        help="Comma-separated expertise gaps (higher=more weight). "
                             "Omit for equal weights.")
    parser.add_argument("--max_questions", type=int, default=None)
    return parser.parse_args()


def load_amateurs(amateur_paths):
    models = []
    for path in amateur_paths:
        print(f"  Loading amateur: {path}")
        m = AutoModelForCausalLM.from_pretrained(
            path.strip(), torch_dtype=torch_dtype, device_map="auto"
        ).eval()
        models.append(m)
    return models


def get_amateur_probs_all(amateur_models, input_ids_batch, tokenizer):
    """Returns [amateur_idx][step_idx] -> prob tensor"""
    all_probs = []
    for am in amateur_models:
        probs = []
        with torch.no_grad():
            for i in range(0, len(input_ids_batch), batch_size):
                chunk  = input_ids_batch[i:i + batch_size]
                padded = torch.nn.utils.rnn.pad_sequence(
                    chunk, batch_first=True, padding_value=tokenizer.pad_token_id
                ).to(device, non_blocking=True)
                mask   = (padded != tokenizer.pad_token_id).long()
                logits = am(padded, attention_mask=mask).logits
                for j, slen in enumerate([x.size(0) for x in chunk]):
                    probs.append(F.softmax(logits[j, slen - 1, :], dim=-1))
        all_probs.append(probs)
    return all_probs


def compute_ensemble_label(expert_dist, amateur_dists, am_weights, alpha, beta):
    """Returns (selected, token_ids, weights, mean_kld) or (False, ...)"""
    eps = 1e-12
    klds = []
    for ap in amateur_dists:
        min_v = min(expert_dist.size(0), ap.size(0))
        ep_ = expert_dist[:min_v] + eps
        ap_ = ap[:min_v] + eps
        klds.append(F.kl_div(ap_.log(), ep_, reduction="sum").item())

    mean_kld = sum(w * k for w, k in zip(am_weights, klds)) / sum(am_weights)
    if mean_kld < beta:
        return False, None, None, mean_kld

    ep_full = expert_dist + eps
    thresh  = alpha * torch.max(ep_full)
    mask    = ep_full >= thresh
    if not mask.any():
        return False, None, None, mean_kld

    log_PE = torch.log(ep_full)
    ensemble_scores = torch.zeros_like(ep_full)
    for ap, w in zip(amateur_dists, am_weights):
        min_v  = min(ep_full.size(0), ap.size(0))
        log_PA = torch.log(ap[:min_v] + eps)
        if min_v < ep_full.size(0):
            pad    = torch.full((ep_full.size(0) - min_v,), -1e9, device=ep_full.device)
            log_PA = torch.cat([log_PA, pad])
        ensemble_scores += w * (log_PE - log_PA)

    scores_V  = ensemble_scores[mask]
    weights_V = torch.softmax(scores_V, dim=-1)
    token_ids = torch.arange(ep_full.size(0), device=ep_full.device)[mask].tolist()
    return True, token_ids, weights_V.tolist(), mean_kld


def main():
    args = parse_args()
    amateur_paths = [p.strip() for p in args.amateur_paths.split(",")]

    if args.amateur_gaps:
        raw_gaps    = [float(g.strip()) for g in args.amateur_gaps.split(",")]
        gap_tensor  = torch.tensor(raw_gaps, dtype=torch.float)
        am_weights  = (gap_tensor / gap_tensor.sum()).tolist()
    else:
        am_weights  = [1.0 / len(amateur_paths)] * len(amateur_paths)

    print(f"\nPrismEnsemble ({len(amateur_paths)} amateurs)")
    for p, w in zip(amateur_paths, am_weights):
        print(f"  {p}  weight={w:.3f}")

    tokenizer     = AutoTokenizer.from_pretrained(expert_model_path)
    expert_model  = AutoModelForCausalLM.from_pretrained(
        expert_model_path, torch_dtype=torch_dtype, device_map="auto"
    ).eval()
    amateur_models = load_amateurs(amateur_paths)

    with open(input_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]
    if args.max_questions:
        prompts = prompts[:args.max_questions]

    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass

    sampled_dataset = []

    for prompt_obj in tqdm(prompts, desc="PrismEnsemble"):
        pid      = prompt_obj["id"]
        question = prompt_obj["question"]
        if pid in processed_ids:
            continue

        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user",   "content": question},
        ]
        text         = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = expert_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1]:].tolist()
        expert_probs   = [F.softmax(s[0], dim=-1) for s in outputs.scores]

        input_ids_batch = []
        for step in range(1, len(expert_targets)):
            prefix   = tokenizer.decode(expert_targets[:step])
            fmt      = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            fmt     += prefix
            ids      = tokenizer(fmt, return_tensors="pt").input_ids[0]
            input_ids_batch.append(ids)

        all_am_probs = get_amateur_probs_all(amateur_models, input_ids_batch, tokenizer)

        count = 0
        for step in range(1, len(expert_targets)):
            ep  = expert_probs[step]
            aps = [all_am_probs[a_idx][step - 1] for a_idx in range(len(amateur_models))]

            selected, tids, wts, mkld = compute_ensemble_label(ep, aps, am_weights, alpha, beta)
            if not selected:
                continue

            sampled_dataset.append({
                "prompt_id":     pid,
                "step":          step,
                "prefix":        tokenizer.decode(expert_targets[:step]),
                "token_ids":     tids,
                "weights":       wts,
                "kl_divergence": mkld,
                "num_amateurs":  len(amateur_models),
            })
            count += 1

        with open(checkpoint_path, "a") as f:
            json.dump({"prompt_id": pid, "num_samples": count}, f)
            f.write("\n")

    with open(output_path, "w") as f:
        for ex in sampled_dataset:
            json.dump(ex, f)
            f.write("\n")

    print(f"\nDone. {len(sampled_dataset)} samples → {output_path}")
    print(f"Compatible with Prism_finetuning.py.")


if __name__ == "__main__":
    main()
