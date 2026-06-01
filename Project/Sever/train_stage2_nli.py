import os
import urllib.request
import json
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

def load_env_token():
    # 1. Try to load from .env file directly
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
                
    # 2. Resolve token (Check HF_API_KEY first, then HF_TOKEN, then environment variables)
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
            data = json.loads(response.read().decode('utf-8'))
            return data.get("name")
    except Exception as e:
        print(f"[WARNING] Hugging Face username을 조회하지 못했습니다: {e}")
        return None

def main():
    print("==================================================")
    print("🚀 Stage 2 Custom NLI Model (xlm-roberta-base) Training Script")
    print("==================================================\n")

    # 1. Resolve Hugging Face Credentials automatically from .env
    hf_token = load_env_token()
    user_id = fetch_hf_username(hf_token)
    model_name = "xlm-roberta-base-nli-stage2"

    if hf_token:
        print(f"✅ .env 파일에서 Hugging Face 토큰을 성공적으로 로드했습니다. ({hf_token[:8]}...)")
        if user_id:
            print(f"✅ Hugging Face 사용자명을 확인했습니다: {user_id}")
            print(f"👉 업로드 경로: https://huggingface.co/{user_id}/{model_name}\n")
        else:
            user_id = "YOUR_HF_USER_ID"
            print("[WARNING] 토큰은 확인되었으나 사용자명을 조회하지 못했습니다.")
            print("Hugging Face에 업로드하려면 사용자명을 수동으로 설정해야 할 수 있습니다.\n")
    else:
        hf_token = "YOUR_HUGGINGFACE_WRITE_TOKEN"
        user_id = "YOUR_HF_USER_ID"
        print("[WARNING] .env 파일이나 환경 변수에서 Hugging Face 토큰(HF_API_KEY)을 찾지 못했습니다.")
        print("토큰 없이 실행 시 모델이 로컬에만 저장됩니다.\n")

    # 2. Download and process the KLUE-NLI dataset
    print("Downloading KLUE NLI dataset from GitHub...")
    train_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_train.json"
    dev_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_dev.json"

    def download_and_load(url):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        df = pd.DataFrame(data)
        
        # Map labels: entailment -> 0, neutral -> 1, contradiction -> 2
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        df['label'] = df['gold_label'].map(label_map)
        
        # Filter out rows with invalid/missing labels
        df = df[df['label'].isin([0, 1, 2])]
        df['label'] = df['label'].astype(int)
        df = df.reset_index(drop=True)
        
        return Dataset.from_pandas(df[['premise', 'hypothesis', 'label']])

    train_dataset = download_and_load(train_url)
    dev_dataset = download_and_load(dev_url)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset
    })
    print(f"Dataset loaded successfully! Train: {len(train_dataset)} samples, Validation: {len(dev_dataset)} samples.")

    # 3. Load Tokenizer & Model (Base-sized model for superior reasoning capability)
    base_model = "xlm-roberta-base"
    print(f"Loading tokenizer and base model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=128
        )
        if "token_type_ids" in tokenized:
            tokenized.pop("token_type_ids")
        return tokenized

    print("Preprocessing dataset...")
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 4. Define evaluation metrics
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # 5. Define Training Arguments
    # Optimized hyper-parameters for klue/roberta-base NLI classification
    should_push = (hf_token != "YOUR_HUGGINGFACE_WRITE_TOKEN" and user_id != "YOUR_HF_USER_ID")
    training_args = TrainingArguments(
        output_dir=model_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,               # Optimal learning rate for base Roberta model
        per_device_train_batch_size=16,   # Adjusted for base-size VRAM footprint
        per_device_eval_batch_size=16,
        num_train_epochs=3,               # 3 Epochs for thorough training of base model
        weight_decay=0.01,
        warmup_ratio=0.1,                 # 10% warmup steps
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=should_push,
        hub_model_id=f"{user_id}/{model_name}" if should_push else None,
        hub_token=hf_token if should_push else None,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Start Training
    print("Starting training...")
    trainer.train()

    # 7. Save and Push model to Hugging Face Hub (if credentials are set)
    print("Saving the best model locally...")
    trainer.save_model(model_name)
    tokenizer.save_pretrained(model_name)
    print(f"Model saved locally to directory: {model_name}/")

    if should_push:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Success! Model uploaded to: https://huggingface.co/{user_id}/{model_name}")
    else:
        print("\n[INFO] Hugging Face Hub에 업로드하려면 .env에 올바른 HF_API_KEY를 입력하고 실행해 주세요.")
        print(f"로컬 저장 완료: {model_name}/")

if __name__ == "__main__":
    main()
