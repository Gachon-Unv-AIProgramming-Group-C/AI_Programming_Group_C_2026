# KLUE-NLI 데이터셋으로 klue/roberta-base 모델을 파인튜닝하는 스크립트 (Stage 2)
# Stage 1의 roberta-small보다 더 큰 모델(111M params)을 사용해 NLI 분류 정확도를 높인다
# 학습된 모델은 nli_server.py의 /nli 엔드포인트에서 사용된다

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
    # .env 파일에서 HuggingFace 토큰을 읽어온다
    # 파일이 여러 위치에 있을 수 있으므로 가능한 경로를 순서대로 시도한다
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
                        # 주석(#)과 빈 줄은 건너뛰고, KEY=VALUE 형태의 줄만 파싱한다
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            env_vars[k.strip()] = v.strip().strip('"').strip("'")
                break
            except Exception:
                pass

    # HF_API_KEY 또는 HF_TOKEN 중 존재하는 값을 우선 사용한다
    token = env_vars.get("HF_API_KEY") or env_vars.get("HF_TOKEN")
    if not token:
        # .env에 없으면 시스템 환경 변수에서도 시도한다
        token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")

    return token

def fetch_hf_username(token):
    # HuggingFace whoami API를 호출하여 토큰에 해당하는 사용자 이름을 반환한다
    # 모델 업로드 경로(username/model_name) 구성에 사용된다
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
    print("Stage 2 NLI Model (klue/roberta-base) Training Script")
    print("==================================================\n")

    # .env에서 HuggingFace 인증 정보를 자동으로 불러온다
    hf_token = load_env_token()
    user_id = fetch_hf_username(hf_token)
    model_name = "klue-roberta-base-nli-stage2"  # HuggingFace Hub에 저장될 모델 이름

    if hf_token:
        print(f".env 파일에서 HuggingFace 토큰을 로드했습니다. ({hf_token[:8]}...)")
        if user_id:
            print(f"HuggingFace 사용자명 확인: {user_id}")
            print(f"업로드 경로: https://huggingface.co/{user_id}/{model_name}\n")
        else:
            user_id = "YOUR_HF_USER_ID"
            print("[WARNING] 사용자명 조회 실패. 수동으로 설정이 필요할 수 있습니다.\n")
    else:
        hf_token = "YOUR_HUGGINGFACE_WRITE_TOKEN"
        user_id = "YOUR_HF_USER_ID"
        print("[WARNING] HF 토큰을 찾지 못했습니다. 모델이 로컬에만 저장됩니다.\n")

    # KLUE-NLI v1.1 데이터셋을 GitHub에서 직접 다운로드한다
    print("Downloading KLUE NLI dataset from GitHub...")
    train_url = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_train.json"
    dev_url   = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark/klue-nli-v1.1/klue-nli-v1.1_dev.json"

    def download_and_load(url):
        # URL에서 JSON을 다운로드하고 HuggingFace Dataset 형식으로 변환한다
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        df = pd.DataFrame(data)

        # 문자열 레이블을 정수 인덱스로 변환한다
        label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        df['label'] = df['gold_label'].map(label_map)

        # 유효하지 않은 레이블(NaN 등)이 포함된 행을 제거한다
        df = df[df['label'].isin([0, 1, 2])]
        df['label'] = df['label'].astype(int)
        df = df.reset_index(drop=True)

        # 모델 입력에 필요한 premise, hypothesis, label 컬럼만 반환한다
        return Dataset.from_pandas(df[['premise', 'hypothesis', 'label']])

    train_dataset = download_and_load(train_url)
    dev_dataset   = download_and_load(dev_url)

    # 학습/검증 데이터셋을 Trainer가 사용할 수 있도록 DatasetDict로 묶는다
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset
    })
    print(f"Dataset loaded. Train: {len(train_dataset)} / Validation: {len(dev_dataset)}")

    # 파인튜닝에 사용할 베이스 모델: klue/roberta-base (한국어 사전학습, 111M params)
    base_model = "klue/roberta-base"
    print(f"Loading tokenizer and base model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 분류 헤드(Linear 768→3)와 레이블 매핑을 포함하여 모델을 초기화한다
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    def preprocess_function(examples):
        # premise와 hypothesis를 쌍으로 토크나이즈한다
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

    # 평가 지표로 정확도를 사용한다
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        # 로짓에서 argmax로 예측 클래스를 선택하고 정확도를 계산한다
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # HuggingFace 토큰이 유효한 경우에만 Hub 업로드를 활성화한다
    should_push = (hf_token != "YOUR_HUGGINGFACE_WRITE_TOKEN" and user_id != "YOUR_HF_USER_ID")

    # 학습 하이퍼파라미터를 설정한다
    training_args = TrainingArguments(
        output_dir=model_name,
        eval_strategy="epoch",                    # 매 에폭 종료 시 검증 수행
        save_strategy="epoch",                    # 매 에폭 체크포인트 저장
        learning_rate=2e-5,                       # 파인튜닝에 적합한 낮은 학습률
        per_device_train_batch_size=16,           # base 모델 크기에 맞게 조정
        per_device_eval_batch_size=16,
        num_train_epochs=3,                       # roberta-base는 3 에폭이 적합
        weight_decay=0.01,                        # L2 정규화
        warmup_ratio=0.1,                         # 전체 스텝의 10%를 학습률 워밍업에 사용
        load_best_model_at_end=True,              # 검증 정확도 최고 시점 모델을 최종 사용
        metric_for_best_model="accuracy",
        push_to_hub=should_push,                  # 토큰이 있으면 학습 완료 후 Hub에 업로드
        hub_model_id=f"{user_id}/{model_name}" if should_push else None,
        hub_token=hf_token if should_push else None,
        report_to="none"                          # 외부 실험 추적 도구 비활성화
    )

    # Trainer 객체를 생성한다. 모델, 학습 설정, 데이터셋, 평가 함수를 하나로 관리한다
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

    # 학습된 모델과 토크나이저를 로컬 디렉터리에 저장한다
    print("Saving the best model locally...")
    trainer.save_model(model_name)
    tokenizer.save_pretrained(model_name)
    print(f"Model saved locally: {model_name}/")

    # 토큰이 유효한 경우 HuggingFace Hub에도 업로드한다
    if should_push:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Success! Model uploaded to: https://huggingface.co/{user_id}/{model_name}")
    else:
        print(f"\n[INFO] Hub 업로드 생략. 로컬 저장 완료: {model_name}/")

if __name__ == "__main__":
    main()
