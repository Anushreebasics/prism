<!-- Icon and title -->

💡 Prism:  
Can <strong><em>SMALL</em></strong> Language Models Teach <strong><em>LARGE</em></strong> Language Models Reasoning?
</h1>



</h3>


<p align="center">
  <strong>Teaching large language models to reason — using smaller ones.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.07962"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square" /></a>
  <a href="https://huggingface.co/collections/bearthecoder/prism-models-68edbf175755ca5a8c699f9c"><img src="https://img.shields.io/badge/🤗-Prism_Models-ffcc4d?style=flat-square" /></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" /></a>
</p>

<br/>

<p align="center">
  <img src="./assets/prism_bars.png" width="720" alt="Prism results overview" />
</p>

---

Most LLM fine-tuning methods treat all tokens the same. Prism doesn't.

The insight behind Prism is simple: during reasoning, not every step is equally hard. A model stumbles at specific moments — and those moments are where learning should be concentrated. Everything else is noise that wastes compute.

To find those moments, Prism uses a small "amateur" model as a reference. Wherever the large "expert" model and the small amateur model strongly disagree on what token comes next, something interesting is happening. The expert knows something the amateur doesn't — and that gap is exactly where fine-tuning should focus.

This leads to a training method that is:
- **99% more token-efficient** than standard SFT
- **90% faster** in total wall-clock time
- **Label-free** — no ground-truth answers needed

And it actually works better.

---

## How It Works

Prism runs in three stages:

**1 — Sample**  
Both the expert and amateur model generate responses to a set of math problems. At each step of the expert's reasoning chain, Prism computes the KL divergence between the two models' next-token distributions. Steps with high divergence are flagged as *informative* — these are the moments the expert genuinely knows something the amateur doesn't.

**2 — Build Supervision Signal**  
At each informative step, Prism constructs a soft training target by contrasting the expert and amateur log-probabilities over plausible tokens. This contrastive distribution captures the expert's advantage in a differentiable form — without requiring the correct answer.

**3 — Fine-tune with LoRA**  
The expert model is fine-tuned using these soft labels via KL divergence loss. Because training only targets a small number of high-signal token positions, the total number of fine-tuning tokens is drastically reduced compared to full-trajectory SFT.

<p align="center">
  <img src="./assets/prism_new.png" width="720" alt="Prism framework diagram" />
</p>

---

## Performance

Trained only on GSM8K, Prism generalizes across 7 math benchmarks.

| Model | GSM8K | MATH | SVAMP | ASDiv | MMLU STEM | Avg |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B (base) | 42.5 | 34.2 | 68.8 | 68.1 | 49.8 | 52.7 |
| + SFT | 69.2 | 57.1 | 64.1 | 70.2 | 47.7 | 61.7 |
| **+ Prism** | **70.6** | **59.3** | **76.0** | **79.8** | **54.9** | **68.1** |
| Qwen2.5-Math-7B (base) | 57.5 | 51.8 | 67.9 | 72.7 | 69.8 | 63.9 |
| + SFT | 64.4 | **63.3** | 76.2 | 76.6 | 68.5 | 69.8 |
| **+ Prism** | **67.9** | 57.8 | **77.2** | **80.6** | **70.5** | **70.8** |
| DeepSeek-R1-Distill-1.5B (base) | 75.2 | 54.2 | 79.9 | 84.9 | 22.3 | 63.3 |
| + SFT | 78.2 | **60.3** | 81.5 | 87.4 | 26.2 | 66.7 |
| **+ Prism** | **79.5** | 60.2 | **83.5** | **87.5** | **26.2** | **67.4** |

---

## Efficiency

| Model | Method | Time | Questions | Tokens | Gain |
|---|---|---|---|---|---|
| Qwen2.5-Math-1.5B | SFT | 4.0 h | 3,952 | 1.77M | +7.7% |
| | **Prism** | **0.5 h** | **1,000** | **0.02M** | **+11.8%** |
| Qwen2.5-Math-7B | SFT | 9.5 h | 6,029 | 2.20M | +4.5% |
| | **Prism** | **0.75 h** | **1,000** | **0.02M** | **+4.7%** |
| DeepSeek-R1-Distill-1.5B | SFT | 3.6 h | 6,023 | 5.95M | +3.0% |
| | **Prism** | **0.5 h** | **1,000** | **0.02M** | **+5.6%** |

<p align="center">
  <img src="./assets/radar_1.5B.png" width="190" />
  <img src="./assets/radar_7B.png" width="190" />
  <img src="./assets/radar_ds1.5B.png" width="190" />
  <img src="./assets/radar_1.5Bins.png" width="187" />
</p>

---

## Quickstart

### Install

```bash
git clone https://github.com/HKUDS/Prism.git
cd Prism
pip install -r requirements.txt
```

### Download Models

You need two models: an expert (domain-strong) and an amateur (general-purpose).

```bash
huggingface-cli download Qwen/Qwen2.5-Math-1.5B --local-dir ./Qwen2.5-Math-1.5B
huggingface-cli download Qwen/Qwen2.5-0.5B      --local-dir ./Qwen2.5-0.5B
```

The expert should clearly outperform the amateur on your target domain. The amateur just needs to be coherent enough to produce meaningful contrast — not accurate. In practice, `Qwen2.5-0.5B` pairs well with the 1.5B and 7B math experts.

### Prepare Data

```bash
python data_prep.py
```

Downloads and formats GSM8K and MATH into JSONL files ready for sampling.

---

## Stage 1 — Sampling

```bash
python Prism_sampling.py --max_questions 1000
```

Open `Prism_sampling.py` and fill in the config block at the top:

```python
expert_model_path  = "./Qwen2.5-Math-1.5B"
amateur_model_path = "./Qwen2.5-0.5B"
device             = "cuda"
torch_dtype        = torch.bfloat16
input_path         = "./gsm8k_train.jsonl"
output_path        = "./prism_samples.jsonl"
checkpoint_path    = "./prism_checkpoint.jsonl"
alpha              = 0.2   # plausibility filter
beta               = 0.4   # KLD selection threshold
batch_size         = 64
```

The script saves a JSONL file where each line is one training instance: a reasoning prefix, the candidate token IDs, and their contrastive weights.

### Skip Sampling — Use Pre-built Samples

If you don't have the GPU budget for sampling, pre-collected datasets are in [`PrismSamples/`](./PrismSamples):

| File | For |
|---|---|
| `LR_Qwen1.5_gsm8k` | Qwen2.5-Math-1.5B |
| `LR_Qwen7_gsm8k` | Qwen2.5-Math-7B |
| `LR_ds1.5_gsm8k` | DeepSeek-R1-Distill-Qwen-1.5B |

Unzip and point `dataset_path` in the fine-tuning script to the extracted JSONL.

---

## Stage 2 — Fine-tuning

```bash
python Prism_finetuning.py
```

For long runs:

```bash
nohup python Prism_finetuning.py > finetune.log 2>&1 &
tail -f finetune.log
```

Config block in `Prism_finetuning.py`:

```python
model_path    = "./Qwen2.5-Math-1.5B"       # must match sampling expert
dataset_path  = "./prism_samples.jsonl"
output_dir    = "./prism-ft-1.5B"
torch_dtype   = torch.bfloat16               # bfloat16 for H100, float16 for A100
batch_size    = 8
max_steps     = 1000
lr            = 5e-5
```

The trainer applies LoRA (`r=8`) to `q_proj` and `v_proj`, then optimizes KL divergence against the contrastive soft labels — no cross-entropy, no full trajectories.

---

## Stage 3 — Merge & Deploy

Fuse the LoRA adapter into the base weights for a standalone model:

```bash
python merge.py
```

#### 📋 Note
Before running the merge script, update the **config section** with your own paths: 

- 🔹 `base_model_path` to your base model directory *(e.g., `./Qwen2.5-Math-7B`)* 

- 🔹 `lora_ckpt_path` to your LoRA checkpoint directory *(e.g., `./ft_qw7_gsm8k/checkpoint-1000`)*  

- 🔹 `merged_model_path` to where you want the merged model to be saved *(e.g., `./ft-7B-merged`)*


---


### 📈 Evaluation

All evaluations are performed using the **official Qwen2.5-Math toolkit**.  

Please refer to the [`evaluation`](./evaluation) folder for detailed usage and setup instructions.


---


## 📊 Main Results

| Model                                         | GSM8K | MATH | SVAMP | ASDiv | Minerva Math | Olympiad Bench | MMLU STEM | AVG. |
|-----------------------------------------------|-------|------|-------|-------|-------------------|---------------|----------------|------|
| **<nobr>Qwen2.5-Math-1.5B</nobr>**            |       |      |       |       |                   |               |                |      |
| Baseline                                      | 42.5  | 34.2 | 68.8  | 68.1  | 9.9               | 23.7          | 49.8           | 42.4 |
| + SFT                                         | 69.2  | 57.1 | 64.1  | 70.2  | **15.1**          | **27.6**      | 47.7           | 50.1 |
| + Prism                                      | **70.6** | **59.3** | **76.0** | **79.8** | 11.4 | 27.1 | **54.9** | **54.2** |
| **<nobr>Qwen2.5-Math-1.5B-Instruct</nobr>**   |       |      |       |       |                   |               |                |      |
| Baseline                                      | 84.8  | 75.8 | 94.2  | 94.7  | 29.4              | 37.5          | 57.4           | 67.7 |
| + SFT                                         | 85.4  | 75.8 | 93.5  | 94.7  | 31.6              | 37.5          | 56.2           | 67.8 |
| + Prism                                      | **86.7** | 75.5 | 93.0 | 94.1 | **32.0** | **37.8** | 55.2 | **67.8** |
| **<nobr>DeepSeek-R1-Distill-Qwen-1.5B</nobr>**|       |      |       |       |                   |               |                |      |
| Baseline                                      | 75.2  | 54.2 | 79.9  | 84.9  | 16.2              | 19.1          | 22.3           | 50.3 |
| + SFT                                         | 78.2  | **60.3** | 81.5 | 87.4 | **18.4** | 21.2 | 26.2 | 53.3 |
| + Prism                                      | **79.5** | 60.2 | **83.5** | **87.5** | 18.0 | **36.5** | **26.2** | **55.9** |
| **<nobr>Qwen2.5-Math-7B</nobr>**              |       |      |       |       |                   |               |                |      |
| Baseline                                      | 57.5  | 51.8 | 67.9  | 72.7  | 14.0              | 16.0          | 69.8           | 50.0 |
| + SFT                                         | 64.4  | **63.3** | 76.2 | 76.6 | 12.1 | **20.5** | 68.5 | 54.5 |
| + Prism                                      | **67.9** | 57.8 | **77.2** | **80.6** | 12.1 | 16.9 | **70.5** | **54.7** |
| **<nobr>Qwen2.5-Math-7B-Instruct</nobr>**     |       |      |       |       |                   |               |                |      |
| Baseline                                      | 95.2  | 83.2 | 93.9  | 95.3  | 33.8              | 41.5          | 69.3           | 73.2 |
| + SFT                                         | 95.4  | 83.1 | **94.1** | 95.2 | **38.2** | 40.7 | 68.2 | **73.6** |
| + Prism                                      | **95.8** | **83.6** | 93.1 | 95.2 | 34.2 | 39.0 | 67.8 | 72.7 |


- Trained *solely* on GSM8K, Prism generalizes effectively for 5 baseline models, achieving consistent gains across 7 benchmarks.

- **+28.1%** on GSM8K, **+25.1%** on MATH, **+7.2%** on SVAMP, **+11.7%** on ASDIV for Qwen2.5-Math-1.5B.  

- **+4.3%** on GSM8K, **+6.0%** on MATH, **+17.4%** on OlympiadBench for DeepSeek-R1-Distill-Qwen-1.5B. 

- **+10.4%** on GSM8K, **+6.0%** on MATH, **+9.3%** on SVAMP, **+7.9%** on ASDIV for Qwen2.5-Math-7B.  

- Efficiency vs. SFT: **90% less total time**, **80% fewer sampled problems**, **99% fewer tuned tokens**.  


---


## ⏱️ Efficiency Study

| **Method** | **Total Time** | **Sampled Problems** | **Tuned Tokens** | **Average Gain** |
|------------|----------|------------|------------|----------|
| **Qwen2.5-Math-1.5B** |||||
| + SFT      | 4.0h     | 3952       | 1.77M      | +7.7%   |
| **+ Prism** | **0.5h** | **1000**  | **0.02M**  | **+11.8%** |
| **Qwen2.5-Math-7B** |||||
| + SFT      | 9.5h     | 6029       | 2.20M      | +4.5%   |
| **+ Prism** | **0.75h** | **1000** | **0.02M**  | **+4.7%** |
| **DeepSeek-R1-Distill-Qwen-1.5B** |||||
| + SFT     | 3.6h     | 6023       | 5.95M      | +3.0%   |
| **+ Prism** | **0.5h** | **1000**  | **0.02M**  | **+5.6%** |
| **Qwen2.5-Math-1.5B-Instruct** |||||
| + SFT     | 3.4h     | 7153       | 2.08M      | +0.1%   |
| **+ Prism** | **0.4h** | **1000**  | **0.02M**  | +0.1%   |


- 🧑‍🏫 **Supervised Fine-Tuning (SFT):** 

  - Implemented with rejection sampling, where models are fine-tuned on demonstrations of correct reasoning trajectories.  
  
  - For a fair comparison, SFT adopts the *same* experimental configuration as Prism, performing LoRA-based fine-tuning *exclusively* on the GSM8K training set.

  - 🎯 **Key Difference:**  
  
    - *Prism* trains on selective next-token predictions, whereas *SFT* optimizes over full reasoning trajectories — an *inherent* difference dictated by their respective training paradigms.  

    - Thus, each *Prism* training instance corresponds to a **single next-token prediction**, whereas each *SFT* example corresponds to a **full reasoning trajectory** comprising a consecutive series of next-token predictions.


- 📈 **Efficiency Evaluation:** 
 
  - ⏱️ **Time Budget** — Sampling time plus fine-tuning time, measured on a *single NVIDIA H200 GPU* without inference accelerators (e.g., vLLM).  
  
  - 📘 **Training Instances** — Number of distinct GSM8K training set problems used to generate the supervision dataset.  
  
  - 🔢 **Tuned Tokens** — Computational overhead measured at the token level.


<p align="center">
  <img src="./assets/radar_1.5B.png" width="200" />
  <img src="./assets/radar_7B.png" width="200" />
  <img src="./assets/radar_ds1.5B.png" width="200" />
  <img src="./assets/radar_1.5Bins.png" width="196" />
  <br>
  <em><strong>Figure 3: Prism matches or surpasses SFT performance with remarkable resource efficiency</strong> — achieving competitive accuracy while cutting training time by 90%, reducing sampled problems by 80%, and requiring 99% fewer tuned tokens.</em>

</p>


💡 **Key Insight:** 

*This marks a fundamental shift in how models are trained — **targeting critical reasoning steps** outperforms brute-force learning, making high-quality AI training achievable even with limited computational resources.*


---


## 🧠 Expertise-Driven Contrast

| **Amateur Model** | **Perf. Gap** | **GSM8K** | **MATH** | **SVAMP** | **ASDiv** | **MMLU STEM** | **AVG.** |
|-------------------|-------------|-----------|----------|-----------|-----------|---------------|----------|
| **Expert: <nobr>Qwen2.5-Math-1.5B</nobr>** |||||||||
| **<nobr>Qwen2.5-0.5B</nobr>**             | **38.2**  | **70.6** | **59.3** | **76.0** | **79.8** | **54.9** | **68.1** |
| <nobr>Qwen2.5-1.5B</nobr>                 | 35.1  | 63.4 | 57.1 | 69.7 | 75.7 | 54.8 | 64.1 |
| <nobr>Qwen2.5-Math-1.5B</nobr>            | /  | / | / | / | / | / | / |
| <nobr>Qwen2.5-Math-1.5B-Ins</nobr>        | -42.3 | 41.4 | 35.5 | 67.5 | 66.4 | 55.0 | 53.2 |
| *Expert Only (Baseline)*                  | /     | 42.5 | 34.2 | 68.8 | 68.1 | 49.8 | 52.7 |
| **Expert: <nobr>Qwen2.5-Math-7B</nobr>** |||||||||
| **<nobr>Qwen2.5-0.5B</nobr>**             | **53.2**  | **67.9** | **57.8** | **77.2** | **80.6** | **70.5** | **70.8** |
| <nobr>Qwen2.5-1.5B</nobr>                 | 50.1  | 69.0 | 56.0 | 77.6 | 78.9 | 69.5 | 70.2 |
| <nobr>Qwen2.5-Math-1.5B</nobr>            | 15.0  | 56.9 | 50.2 | 63.5 | 63.4 | 70.7 | 60.9 |
| <nobr>Qwen2.5-Math-1.5B-Ins</nobr>        | -27.3 | 59.4 | 49.0 | 68.3 | 69.6 | 70.3 | 63.3 |
| *Expert Only (Baseline)*                  | /     | 57.5 | 51.8 | 67.9 | 72.7 | 69.8 | 63.9 |


- **Domain Expertise over Scale:** *The success of Expert–Amateur collaboration is driven most effectively by domain-specific knowledge rather than model size (e.g., Qwen2.5-Math-1.5B vs. Qwen2.5-1.5B), freeing Prism from rigid scaling constraints.*

- **Dependence on Expertise Gap:** *Performance gains are closely correlated with the size of the expertise gap — as the Amateur approaches the Expert’s capability, contrastive signals weaken and improvements diminish.*


---

## 🔍 More Insights

<p align="center">
  <img src="./assets/gap_vs_perf.png" alt="Sampling Stage" width="55.5%"/>
  <img src="./assets/radar_ablations.png" alt="Fine-tuning Stage" width="34.5%"/>
</p>

<p align="center">
  
  <em>👈 Figure 4(a): Expert–Amateur Pairing Effects — Each point represents a fixed Expert model paired with an Amateur model. The performance gains achieved by Prism diminish as the expertise gap narrows.</em><br>

  <em>👉 Figure 4(b): Impact of Ablation — Removing key components from Prism consistently reduces performance, revealing their critical contributions.</em>

</p>


---


## 🏆 Comparison with Competing Methods

<table>
<tr>
<td>

<!-- Left Table -->
  
| **Attribute**        | **Time** | **SFT** | **Prism** |
|-----------------------|----------------|---------|------------|
| Full trajectories     | ⬆️          | ✅      | ❌         |
| All-token tuning      | ⬆️          | ✅      | ❌         |
| Prefix termination    | ⬇️          | ❌      | ✅         |
| Selective tokens      | ⬇️          | ❌      | ✅         |
| Verification-free     | ⬇️          | ❌      | ✅         |

</td>
<td>

<!-- Right Table -->

| **Attribute**         | **Utility** | **CD**      | **Prism** |
|------------------------|------------------|-------------|------------|
| Contrast usage         | /                | Inference   | Training   |
| Size-based contrast    | ⬇️            | ✅          | ❌         |
| Expertise contrast     | ⬆️            | ❌          | ✅         |
| Persistent benefits    | ⬆️            | ❌          | ✅         |
| Standalone inference  | ⬆️            | ❌          | ✅         |

</td>
</tr>
</table>

- 👈 *Left:* Efficiency contrasts at a glance. ⬆️ and ⬇️ indicate whether each aspect helps or hurts the overall efficiency of the method. 
  
- 👉 *Right:* Key differences between traditional Contrastive Decoding (CD) methods and Prism. ⬆️ and ⬇️ indicate whether each aspect helps or hurts the practicality of the method.


---




