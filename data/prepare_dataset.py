"""
Dataset preparation utilities for QLoRA Domain Specialist.

Supports:
1. Public Hugging Face datasets with chat format
2. Custom JSONL files (instruction / input / output or messages format)
"""

from datasets import load_dataset, Dataset
from typing import Optional, List, Dict
import json
from pathlib import Path


def load_public_dataset(
    name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft[:4000]",
) -> Dataset:
    """Load a public instruction / chat dataset from Hugging Face Hub."""
    print(f"Loading dataset: {name} | split: {split}")
    ds = load_dataset(name, split=split)
    print(f"Loaded {len(ds)} examples")
    return ds


def load_custom_jsonl(path: str) -> Dataset:
    """
    Load a custom JSONL file.

    Expected formats (auto-detected):
    A) Chat format:
       {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    B) Alpaca / Dolly style:
       {"instruction": "...", "input": "...", "output": "..."}
       or {"instruction": "...", "response": "..."}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} custom examples from {path}")
    return Dataset.from_list(records)


def convert_alpaca_to_messages(example: Dict) -> Dict:
    """Convert classic Alpaca/Dolly format into chat messages."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "") or example.get("context", "")
    output = example.get("output", "") or example.get("response", "")

    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def prepare_dataset(
    dataset_name: Optional[str] = None,
    split: str = "train_sft[:4000]",
    custom_path: Optional[str] = None,
) -> Dataset:
    """
    Main entry point.
    Prefer custom_path if provided, otherwise load public dataset.
    """
    if custom_path:
        ds = load_custom_jsonl(custom_path)
        # Convert if needed
        if "messages" not in ds.column_names and "instruction" in ds.column_names:
            ds = ds.map(convert_alpaca_to_messages, remove_columns=ds.column_names)
        return ds

    if dataset_name is None:
        dataset_name = "HuggingFaceH4/ultrachat_200k"

    return load_public_dataset(dataset_name, split)


def create_sample_custom_data(output_path: str = "data/sample_domain.jsonl"):
    """
    Creates a tiny sample domain dataset (Customer Support style)
    so users can test the pipeline immediately.
    """
    samples = [
        {
            "messages": [
                {"role": "user", "content": "My order #4521 has not arrived yet. It has been 8 days."},
                {"role": "assistant", "content": "I'm sorry for the delay with order #4521. Let me check the status for you. According to our records, the package is currently in transit and expected to arrive within 2 business days. Would you like me to send you the tracking link?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": "You can reset your password by following these steps:\n1. Go to the login page\n2. Click on 'Forgot Password'\n3. Enter the email associated with your account\n4. Check your inbox for a reset link (valid for 30 minutes)\nIf you don't receive the email, please check your spam folder or contact support."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is your refund policy?"},
                {"role": "assistant", "content": "We offer a 30-day money-back guarantee on all products. To request a refund:\n1. Log into your account\n2. Go to Orders → Select the order\n3. Click 'Request Refund'\nRefunds are processed within 5-7 business days after we receive the returned item."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "The product I received is damaged."},
                {"role": "assistant", "content": "I'm very sorry that you received a damaged product. Please follow these steps so we can resolve this quickly:\n1. Take clear photos of the damage\n2. Reply with the photos and your order number\n3. We will arrange a free replacement or full refund within 24 hours.\nThank you for your patience."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Do you offer international shipping?"},
                {"role": "assistant", "content": "Yes, we ship to over 80 countries. Shipping times and costs vary by destination:\n• North America / Europe: 5-10 business days\n• Asia / Australia: 7-14 business days\n• Rest of World: 10-20 business days\nYou can see exact shipping costs at checkout after entering your address."}
            ]
        },
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Sample domain dataset written to {path}")
    return str(path)


if __name__ == "__main__":
    # Create a small sample so the project is immediately runnable
    create_sample_custom_data()
    print("Dataset preparation utilities ready.")
