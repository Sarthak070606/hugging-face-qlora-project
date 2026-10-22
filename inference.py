#!/usr/bin/env python3
"""
Simple inference script for the QLoRA Domain Specialist.
Loads the base model + adapter and runs interactive or single-prompt generation.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(cfg: dict):
    model_id = cfg["model"]["base_model"]
    adapter_path = cfg["training"]["output_dir"]

    print(f"Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if Path(adapter_path).exists():
        print(f"Loading QLoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    else:
        print(f"[Warning] No adapter found at {adapter_path}. Using base model only.")

    return model, tokenizer


def chat(model, tokenizer, user_message: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": user_message}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"User: {user_message}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Try to extract only the new part
    if prompt in full_response:
        return full_response[len(prompt):].strip()
    return full_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(cfg)

    if args.prompt:
        response = chat(model, tokenizer, args.prompt)
        print("\nAssistant:", response)
        return

    print("\nInteractive chat (type 'quit' or 'exit' to stop)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if not user_input:
            continue
        response = chat(model, tokenizer, user_input)
        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    main()
