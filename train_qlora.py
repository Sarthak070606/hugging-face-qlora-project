#!/usr/bin/env python3
"""
QLoRA Domain Specialist - Training Script
=========================================
Fine-tunes an open LLM using QLoRA (4-bit + LoRA) with the modern
Hugging Face stack: transformers + peft + bitsandbytes + trl.

Usage:
    python train_qlora.py
    python train_qlora.py --config config.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# Ensure project root is on sys.path (fixes Windows import issues)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_dataset import prepare_dataset, create_sample_custom_data


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_trainable_parameters(model):
    """Show how many parameters are actually being trained (key QLoRA metric)."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    print("=" * 60)
    print("QLoRA Domain Specialist - Training")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Tokenizer
    # ----------------------------------------------------------
    model_id = cfg["model"]["base_model"]
    print(f"\nLoading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ----------------------------------------------------------
    # 2. 4-bit Quantization (QLoRA)
    # ----------------------------------------------------------
    print("\nConfiguring 4-bit quantization (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )

    # ----------------------------------------------------------
    # 3. Load base model in 4-bit
    # ----------------------------------------------------------
    print(f"Loading base model in 4-bit: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,  # newer transformers prefer `dtype` over `torch_dtype`
        trust_remote_code=True,
        attn_implementation="eager",  # safe default; change to flash_attention_2 if available
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing

    # ----------------------------------------------------------
    # 4. LoRA / QLoRA adapter
    # ----------------------------------------------------------
    print("\nApplying LoRA adapters...")
    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )

    # ----------------------------------------------------------
    # 5. Dataset
    # ----------------------------------------------------------
    print("\nPreparing dataset...")
    # Create a tiny sample if user wants to test without downloading large data
    sample_path = Path("data/sample_domain.jsonl")
    if not sample_path.exists():
        create_sample_custom_data(str(sample_path))

    # Prefer the small sample for first run so the project works offline-ish
    # Switch to public dataset by changing the next line
    use_sample = True  # set to False to use the public UltraChat subset

    if use_sample:
        print("Using local sample domain dataset (Customer Support style)")
        dataset = prepare_dataset(custom_path=str(sample_path))
        # For demo purposes we repeat the small set a few times
        dataset = dataset.shuffle(seed=42)
    else:
        dataset = prepare_dataset(
            dataset_name=cfg["dataset"]["name"],
            split=cfg["dataset"]["split"],
        )

    print(f"Training examples: {len(dataset)}")

    # ----------------------------------------------------------
    # 6. Training arguments
    # ----------------------------------------------------------
    output_dir = cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compatible with both older and newer TRL versions
    # (some versions reject warmup_ratio / certain keys)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_steps=5,                    # safer than warmup_ratio across TRL versions
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        bf16=cfg["training"]["bf16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        max_length=cfg["model"]["max_seq_length"],  # newer TRL uses max_length
        packing=cfg["training"]["packing"],
        report_to=cfg["training"]["report_to"],
        optim="paged_adamw_8bit",
        logging_first_step=True,
    )

    # ----------------------------------------------------------
    # 7. Trainer
    # ----------------------------------------------------------
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Show trainable parameter count (very important for portfolio)
    print("\n--- Parameter Efficiency ---")
    print_trainable_parameters(trainer.model)

    # ----------------------------------------------------------
    # 8. Train
    # ----------------------------------------------------------
    print("\nStarting QLoRA training...")
    trainer.train()

    # ----------------------------------------------------------
    # 9. Save
    # ----------------------------------------------------------
    print(f"\nSaving adapter + tokenizer to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Optional: push to Hub
    if cfg.get("hub", {}).get("push_to_hub", False):
        repo_id = cfg["hub"]["repo_id"]
        print(f"Pushing to Hugging Face Hub: {repo_id}")
        trainer.model.push_to_hub(repo_id, private=cfg["hub"].get("private", False))
        tokenizer.push_to_hub(repo_id, private=cfg["hub"].get("private", False))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapter saved at: {output_dir}")
    print("Next steps:")
    print("  1. python evaluate.py")
    print("  2. python inference.py")
    print("  3. python app.py          # Gradio demo")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Domain Specialist Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
