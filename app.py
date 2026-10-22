"""
Gradio Demo for QLoRA Domain Specialist.
Run after training: python app.py
"""

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from pathlib import Path
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Global model and tokenizer
model = None
tokenizer = None
cfg = None


def load_resources():
    global model, tokenizer, cfg

    cfg = load_config()
    model_id = cfg["model"]["base_model"]
    adapter_path = cfg["training"]["output_dir"]

    print(f"Loading model: {model_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if Path(adapter_path).exists():
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        raise FileNotFoundError(
            f"Trained adapter not found: {adapter_path}\n"
            "Check training.output_dir in config.yaml."
        )

    model.eval()
    print("Model ready.")


def respond(message, history, max_tokens, temperature):
    if model is None or tokenizer is None:
        return "Model not loaded. Check the terminal for errors."

    # Include conversation history so the chatbot can remember earlier turns.
    messages = []

    for item in history or []:
        if isinstance(item, dict):
            if item.get("role") in ("user", "assistant") and item.get("content"):
                messages.append({
                    "role": item["role"],
                    "content": item["content"],
                })
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            if user_message:
                messages.append({
                    "role": "user",
                    "content": user_message,
                })
            if assistant_message:
                messages.append({
                    "role": "assistant",
                    "content": assistant_message,
                })

    messages.append({"role": "user", "content": message})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = "\n".join(
            f"{item['role'].capitalize()}: {item['content']}"
            for item in messages
        ) + "\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(
        model.get_input_embeddings().weight.device
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens, not the original prompt.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return reply or "I couldn't generate a response. Please try again."


def build_demo():
    load_resources()

    with gr.Blocks(title="QLoRA Domain Specialist") as demo:
        gr.Markdown(
            """
            # QLoRA Domain Specialist

            Fine-tuned using **QLoRA (4-bit quantization + LoRA)**,
            Hugging Face Transformers, and PEFT.

            Ask questions to test your trained domain-specific model.
            """
        )

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[
                gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=256,
                    step=32,
                    label="Max new tokens",
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=1.2,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                ),
            ],
            examples=[
                [
                    "My order #4521 has not arrived yet. It has been 8 days.",
                    256,
                    0.7,
                ],
                ["How do I reset my password?", 256, 0.7],
                ["What is your refund policy?", 256, 0.7],
                ["The product I received is damaged.", 256, 0.7],
                ["Do you offer international shipping?", 256, 0.7],
            ],
            title="Chat with your fine-tuned model",
        )

        gr.Markdown(
            f"""
            ---
            **Base model:** `{cfg['model']['base_model']}`

            **Adapter path:** `{cfg['training']['output_dir']}`

            **Technique:** QLoRA (NF4 + LoRA r={cfg['lora']['r']})
            """
        )

        gr.Markdown(
            "<center>Made with ❤️ by Sarthak Jain</center>"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
