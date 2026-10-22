# QLoRA Domain Specialist

**Parameter-Efficient Domain Adaptation of LLMs using QLoRA**  
*(Hugging Face Transformers + PEFT + bitsandbytes + TRL)*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

This project demonstrates **end-to-end mastery** of modern parameter-efficient fine-tuning:

| Technique | Description |
|-----------|-------------|
| **QLoRA** | 4-bit NF4 quantization + Low-Rank Adaptation |
| **PEFT** | Train only ~0.1–1% of parameters |
| **TRL SFTTrainer** | Production-grade supervised fine-tuning |
| **bitsandbytes** | Memory-efficient quantization |
| **Hugging Face Hub** | Model & adapter publishing |

**Why this project is strong for a portfolio:**
- Shows you understand memory constraints of LLMs
- Uses the current industry-standard stack (2025–2026)
- Includes training → evaluation → inference → Gradio demo
- Clear before/after comparison
- Easy to extend to any domain (medical, code, support, legal…)

---

## Project Structure

```
qlora-domain-specialist/
├── README.md
├── requirements.txt
├── config.yaml                 # All hyperparameters in one place
├── train_qlora.py              # Main QLoRA training script
├── evaluate.py                 # Base model vs Fine-tuned comparison
├── inference.py                # CLI chat / single prompt
├── app.py                      # Gradio web demo
├── data/
│   ├── prepare_dataset.py      # Dataset loading utilities
│   └── sample_domain.jsonl     # Tiny customer-support sample (auto-created)
├── notebooks/                  # (optional) exploration notebooks
└── outputs/                    # Saved adapters appear here after training
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `bitsandbytes` requires a CUDA GPU.  
> Free options: Google Colab (T4), Kaggle, or any cloud GPU with ≥ 8–12 GB VRAM.

### 2. (Optional) Edit configuration

Open `config.yaml` and change:
- Base model
- LoRA rank (`r`)
- Dataset size / path
- Learning rate, epochs, etc.

### 3. Train with QLoRA

```bash
python train_qlora.py
```

By default the script uses a small **Customer Support** sample dataset so you can verify the full pipeline quickly.  
To use a larger public dataset (UltraChat), open `train_qlora.py` and set `use_sample = False`.

### 4. Evaluate (Before vs After)

```bash
python evaluate.py
```

### 5. Interactive inference

```bash
python inference.py
# or single prompt
python inference.py --prompt "How do I reset my password?"
```

### 6. Launch Gradio demo

```bash
python app.py
```

Then open `http://localhost:7860`.

---

## Key Technical Highlights

### QLoRA Configuration
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### LoRA Configuration
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",   # modern best practice
)
```

### Trainable Parameters
After training you will see something similar to:
```
trainable params: 4,194,304 || all params: 1,778,000,000 || trainable%: 0.2359
```
→ You are updating **less than 0.3%** of the model.

---

## Recommended Hardware

| Model | Method | Approx. VRAM |
|-------|--------|--------------|
| Qwen2.5-1.5B | QLoRA | ~6–8 GB |
| Phi-3-mini / Qwen2.5-3B | QLoRA | ~10–14 GB |
| Llama-3.2-3B | QLoRA | ~12–16 GB |
| 7B-class models | QLoRA | ~16–24 GB |

---

## Extending to Your Own Domain

1. Create a JSONL file in `data/my_domain.jsonl` using either format:

**Chat format (preferred):**
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Alpaca style:**
```json
{"instruction": "...", "input": "...", "output": "..."}
```

2. In `train_qlora.py` change the dataset loading section to use your file.

3. Re-train and evaluate.

---

## Pushing to Hugging Face Hub

1. Login:
```bash
huggingface-cli login
```

2. In `config.yaml` set:
```yaml
hub:
  push_to_hub: true
  repo_id: "your-username/qwen2.5-1.5b-qlora-domain"
```

3. Run training again (or push manually after training).

---

## What This Demonstrates on a Resume / Interview

- Practical knowledge of **LoRA** and **QLoRA**
- Ability to fine-tune modern LLMs under memory constraints
- End-to-end ML engineering (data → train → eval → serve)
- Clean configuration management and reproducible experiments
- Familiarity with the full Hugging Face ecosystem

---

## License

MIT License – feel free to use this project as a starting point for your own domain adapters.

---

**Author:** Sarthak Jain  
**Stack:** Hugging Face Transformers • PEFT • TRL • bitsandbytes • Gradio
