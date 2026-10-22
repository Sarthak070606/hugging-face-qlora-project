#!/usr/bin/env python3
"""
Before vs After evaluation for QLoRA Domain Specialist.
Compares base model responses with the fine-tuned adapter on the same prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


TEST_PROMPTS = [
    "My order #4521 has not arrived yet. It has been 8 days.",
    "How do I reset my password?",
    "What is your refund policy?",
    "The product I received is damaged.",
    "Do you offer international shipping?",
]


def generate(pipe, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    # Many instruct models expect the chat template
    try:
        text = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"User: {prompt}\nAssistant:"

    outputs = pipe(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    generated = outputs[0]["generated_text"]
    # Return only the assistant part when possible
    if "Assistant:" in generated:
        return generated.split("Assistant:")[-1].strip()
    return generated[len(text):].strip()


def main():
    cfg = load_config()
    model_id = cfg["model"]["base_model"]
    adapter_path = cfg["training"]["output_dir"]

    print("=" * 60)
    print("QLoRA Evaluation – Base Model vs Fine-tuned Adapter")
    print("=" * 60)

    # ---- Base model (4-bit) ----
    print("\nLoading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
    )

    # ---- Fine-tuned adapter ----
    adapter_exists = Path(adapter_path).exists() and any(Path(adapter_path).glob("*.safetensors")) or any(Path(adapter_path).glob("adapter_model*"))

    if adapter_exists:
        print(f"Loading fine-tuned adapter from {adapter_path}...")
        ft_model = PeftModel.from_pretrained(base_model, adapter_path)
        ft_pipe = pipeline(
            "text-generation",
            model=ft_model,
            tokenizer=tokenizer,
        )
    else:
        print(f"\n[Warning] Adapter not found at {adapter_path}")
        print("Run train_qlora.py first. Showing only base model responses.\n")
        ft_pipe = None

    # ---- Compare ----
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─'*60}")
        print(f"PROMPT {i}: {prompt}")
        print(f"{'─'*60}")

        print("\n[BASE MODEL]")
        base_out = generate(base_pipe, prompt)
        print(base_out[:600] + ("..." if len(base_out) > 600 else ""))

        if ft_pipe is not None:
            print("\n[FINE-TUNED QLoRA]")
            ft_out = generate(ft_pipe, prompt)
            print(ft_out[:600] + ("..." if len(ft_out) > 600 else ""))

    print("\n" + "=" * 60)
    print("Evaluation finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
