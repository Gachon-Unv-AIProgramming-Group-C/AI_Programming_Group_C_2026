# -*- coding: utf-8 -*-
"""
Google Colab용 Custom NLI Fine-tuning 스크립트
이 코드를 Google Colab에 복사하여 GPU(T4) 환경에서 실행하세요.
"""

# 1. 필수 라이브러리 설치
# !pip install transformers datasets evaluate accelerate huggingface_hub pandas

import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# 2. Hugging Face 로그인 및 설정
# Hugging Face Access Token (Write 권한 필요)을 입력하세요.
# https://huggingface.co/settings/tokens 에서 토큰을 생성할 수 있습니다.
HF_TOKEN = "YOUR_HUGGINGFACE_WRITE_TOKEN" 
USER_ID = "YOUR_HUGGINGFACE_USERNAME"
MODEL_NAME = "klue-roberta-small-nli" # Hugging Face에 등록될 모델 이름

if HF_TOKEN == "YOUR_HUGGINGFACE_WRITE_TOKEN":
    print("[WARNING] HF_TOKEN을 발급받아 입력해주세요! 모델을 업로드하기 위해 필수적입니다.")

# 3. 데이터셋 로드 (Hugging Face Datasets API 버그 우회를 위해 GitHub에서 직접 JSON 다운로드)
import urllib.request
import json
import pandas as pd
from datasets import Dataset, DatasetDict

print("Downloading KLUE NLI dataset from GitHub...")
train_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_train.json"
dev_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_dev.json"

def download_and_load(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
    df = pd.DataFrame(data)
    
    # gold_label -> 정수형 label 변환 (entailment: 0, neutral: 1, contradiction: 2)
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    df['label'] = df['gold_label'].map(label_map)
    
    # NaN 결측치 및 0, 1, 2 이외의 이상치 데이터 완전히 제외
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
print("Dataset loaded successfully!")

# 4. 토크나이저 및 모델 정의
base_model = "klue/roberta-small"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# KLUE NLI는 entailment(0), neutral(1), contradiction(2) 3개의 라벨을 가집니다.
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# 5. 데이터셋 전처리 함수
def preprocess_function(examples):
    # Premise와 Hypothesis를 합쳐서 모델의 입력 포맷으로 변환
    tokenized = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=128
    )
    # klue/roberta 모델은 segment embedding size가 1이므로 token_type_ids(0과 1)가 있으면 에러가 납니다.
    # 따라서 이 필드를 제거해 줍니다.
    if "token_type_ids" in tokenized:
        tokenized.pop("token_type_ids")
    return tokenized

print("Preprocessing dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 6. 평가 지표 정의 (Accuracy)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)

# 7. 학습 인자(Training Arguments) 설정
training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2, # 빠르게 학습하기 위해 2 Epoch 설정
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True, # 학습 완료 후 자동으로 Hugging Face Hub에 업로드
    hub_model_id=f"{USER_ID}/{MODEL_NAME}",
    hub_token=HF_TOKEN,
    report_to="none"
)

# 8. Trainer 선언 및 학습 개시
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# 9. 최종 모델을 Hugging Face Hub에 명시적 업로드
print("Pushing final model to Hugging Face Hub...")
trainer.push_to_hub()
print(f"Success! Model uploaded to: https://huggingface.co/{USER_ID}/{MODEL_NAME}")
