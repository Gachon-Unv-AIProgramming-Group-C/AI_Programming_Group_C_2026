import os
import urllib.request
import re
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

def load_env_token():
    env_vars = {}
    possible_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "Project", "Sever", ".env"),
        ".env"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip().strip('"').strip("'")
                break
            except Exception:
                pass
    token = env_vars.get("HF_API_KEY") or env_vars.get("HF_TOKEN")
    if not token:
        token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    return token

def fetch_hf_username(token):
    if not token or token == "YOUR_HUGGINGFACE_WRITE_TOKEN":
        return None
    url = "https://huggingface.co/api/whoami"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            import json
            data = json.loads(response.read().decode('utf-8'))
            return data.get("name")
    except Exception as e:
        print(f"[WARNING] Hugging Face username을 조회하지 못했습니다: {e}")
        return None

def main():
    print("==================================================")
    print("🚀 Local Question Paraphraser Training Script (Qwen SFT)")
    print("==================================================\n")

    hf_token = load_env_token()
    user_id = fetch_hf_username(hf_token)
    model_dir = "local-qwen-paraphraser"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Download KakaoBrain ParaKQC dataset
    print("Downloading KakaoBrain ParaKQC dataset...")
    url = "https://raw.githubusercontent.com/warnikchow/paraKQC/master/data/paraKQC_v1.txt"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    lines = []
    with urllib.request.urlopen(req) as response:
        for line in response:
            lines.append(line.decode('utf-8').strip())

    label_to_sentences = {}
    valid_lines = [line for line in lines if line]
    for index, line in enumerate(valid_lines):
        parts = line.split('\t')
        if len(parts) == 3:
            idx, label, sentence = parts
            group_id = index // 10
            if group_id not in label_to_sentences:
                label_to_sentences[group_id] = []
            label_to_sentences[group_id].append(sentence)
    
    num_sentences = sum(len(s) for s in label_to_sentences.values())
    print(f"Loaded {num_sentences} sentences across {len(label_to_sentences)} unique paraphrase groups.")

    # Create paraphrase pairs
    print("Generating sentence pairs...")
    pairs = []
    for group_id, sentences in label_to_sentences.items():
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    pairs.append({"input": sentences[i], "target": sentences[j]})

    import random
    random.seed(42)
    random.shuffle(pairs)
    
    # We use 5000 pairs to make training fast and fit in memory
    pairs = pairs[:5000]
    print(f"Generated {len(pairs)} training pairs.")

    split_idx = 4500
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    # 2. Load Qwen Tokenizer & Model
    print(f"Loading tokenizer and model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Formatting chat template for SFT training
    system_prompt = "You are a question paraphrasing expert. Generate exactly 1 semantic variation of the given question."

    def preprocess_function(examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for inp, tgt in zip(examples["input"], examples["target"]):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: \"{inp}\""},
                {"role": "assistant", "content": tgt}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(text, max_length=256, truncation=True)
            
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            
            # Mask the prompt labels so we only calculate loss on the target response
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: \"{inp}\""}
            ]
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompt_tokenized = tokenizer(prompt_text)
            prompt_len = len(prompt_tokenized["input_ids"])

            labels = [-100] * prompt_len + input_ids[prompt_len:]
            
            # Pad sequences to max length
            padding_len = 256 - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                labels = labels + [-100] * padding_len
            else:
                input_ids = input_ids[:256]
                attention_mask = attention_mask[:256]
                labels = labels[:256]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }

    print("Preprocessing datasets...")
    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])
    val_tokenized = val_dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])

    # 3. Training Arguments
    should_push = (hf_token is not None and user_id is not None)
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,   # Reduced for 6GB VRAM compatibility
        per_device_eval_batch_size=2,    # Reduced for 6GB VRAM compatibility
        gradient_accumulation_steps=4,   # Keeps effective batch size at 8
        fp16=True,                       # Enables mixed precision (safe since model loaded in float32)
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=1,  # 1 epoch is fast and sufficient for SFT on 0.5B model
        logging_steps=50,
        push_to_hub=should_push,
        hub_model_id=f"{user_id}/{model_dir}" if should_push else None,
        hub_token=hf_token if should_push else None,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
    )

    # 4. Train
    print("Training Qwen paraphraser model...")
    trainer.train()

    print("Saving the best model locally...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}/")

    if should_push:
        print("Pushing to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Success! Model uploaded to: https://huggingface.co/{user_id}/{model_dir}")

if __name__ == "__main__":
    main()
