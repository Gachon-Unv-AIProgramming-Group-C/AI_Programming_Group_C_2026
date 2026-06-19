# KLUE-NLI 데이터셋으로 klue/roberta-small 모델을 파인튜닝하는 스크립트 (Stage 1)
# 전제(premise)와 가설(hypothesis)이 주어졌을 때 entailment/neutral/contradiction을 분류한다

# !pip install transformers datasets evaluate accelerate huggingface_hub pandas

import os
import urllib.request
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# HuggingFace Hub 업로드에 필요한 토큰과 사용자 정보
HF_TOKEN = "..."
USER_ID = "..."
MODEL_NAME = "klue-roberta-small-nli"  # 저장될 모델 이름

if HF_TOKEN == "YOUR_HUGGINGFACE_WRITE_TOKEN":
    print("[WARNING] HF_TOKEN을 발급받아 입력해주세요! 모델을 업로드하기 위해 필수적입니다.")

# KLUE-NLI v1.1 데이터셋 URL (GitHub 공개 데이터)
print("Downloading KLUE NLI dataset from GitHub...")
train_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_train.json"
dev_url   = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_dev.json"

def download_and_load(url):
    # HTTP 요청으로 JSON 파일을 다운로드하고 DataFrame으로 변환한다
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
    df = pd.DataFrame(data)

    # 문자열 레이블을 정수 인덱스로 변환 (entailment=0, neutral=1, contradiction=2)
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    df['label'] = df['gold_label'].map(label_map)

    # 매핑되지 않은 레이블(NaN)이 있는 행을 제거한다
    df = df[df['label'].isin([0, 1, 2])]
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)

    # 모델 학습에 필요한 세 컬럼만 남기고 HuggingFace Dataset 형식으로 반환
    return Dataset.from_pandas(df[['premise', 'hypothesis', 'label']])


# 학습 및 검증 데이터셋을 다운로드한다
train_dataset = download_and_load(train_url)
dev_dataset   = download_and_load(dev_url)

# HuggingFace DatasetDict로 묶어 Trainer가 사용할 수 있게 한다
dataset = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset
})
print("Dataset loaded successfully!")

# 파인튜닝에 사용할 베이스 모델을 지정한다 (한국어 사전학습 RoBERTa Small)
base_model = "klue/roberta-small"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# 인덱스 ↔ 레이블 이름 매핑 정보를 모델에 등록한다
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

# 분류 헤드(Linear 768→3)가 추가된 시퀀스 분류 모델을 불러온다
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

def preprocess_function(examples):
    # premise와 hypothesis를 하나의 입력 시퀀스로 토크나이즈한다
    # 형식: [CLS] premise [SEP] hypothesis [SEP]
    tokenized = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=128     # 128 토큰을 초과하는 입력은 잘라낸다
    )
    # RoBERTa는 token_type_ids를 사용하지 않으므로 제거한다
    if "token_type_ids" in tokenized:
        tokenized.pop("token_type_ids")
    return tokenized

print("Preprocessing dataset...")
# 전체 데이터셋에 토크나이즈 함수를 일괄 적용한다
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 평가 지표로 정확도(Accuracy)를 사용한다
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    # 로짓에서 가장 높은 값의 인덱스를 예측 클래스로 선택한다
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)

# 학습 하이퍼파라미터를 설정한다
training_args = TrainingArguments(
    output_dir=MODEL_NAME,                   # 체크포인트 저장 경로
    eval_strategy="epoch",                   # 매 에폭마다 검증 수행
    save_strategy="epoch",                   # 매 에폭마다 체크포인트 저장
    learning_rate=2e-5,                      # 파인튜닝에 적합한 낮은 학습률
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,                      # 빠른 학습을 위해 2 에폭으로 설정
    weight_decay=0.01,                       # L2 정규화로 과적합 방지
    load_best_model_at_end=True,             # 검증 정확도가 가장 높은 체크포인트를 최종 모델로 사용
    metric_for_best_model="accuracy",
    push_to_hub=True,                        # 학습 완료 후 HuggingFace Hub에 자동 업로드
    hub_model_id=f"{USER_ID}/{MODEL_NAME}",
    hub_token=HF_TOKEN,
    report_to="none"                         # wandb 등 외부 로깅 비활성화
)

# Trainer를 초기화한다. 모델, 설정, 데이터, 평가함수를 한 번에 관리한다
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()  # 파인튜닝 시작

# 학습된 모델을 HuggingFace Hub에 업로드한다
print("Pushing final model to Hugging Face Hub...")
trainer.push_to_hub()
print(f"Success! Model uploaded to: https://huggingface.co/{USER_ID}/{MODEL_NAME}")
