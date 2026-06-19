# 한국어 질문 패러프레이저를 Qwen 0.5B 모델에 SFT(지도 파인튜닝)로 학습하는 스크립트
# KakaoBrain ParaKQC 데이터셋을 사용해 동일 의미의 질문을 다르게 표현하는 능력을 학습시킨다
# 학습된 모델은 nli_server.py의 /paraphrase 엔드포인트에서 사용된다 (Layer 3 SAC³)

import os
import urllib.request
import re
import random
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)

def load_env_token():
    # .env 파일에서 HuggingFace 토큰을 읽어온다
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
    # HF_API_KEY 우선, 없으면 HF_TOKEN, 없으면 환경 변수에서 시도
    token = env_vars.get("HF_API_KEY") or env_vars.get("HF_TOKEN")
    if not token:
        token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    return token

def fetch_hf_username(token):
    # HuggingFace API로 토큰에 해당하는 사용자 이름을 조회한다
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
    print("Qwen 0.5B 한국어 패러프레이저 SFT 학습 스크립트")
    print("==================================================\n")

    # .env에서 HuggingFace 인증 정보를 불러온다
    hf_token = load_env_token()
    user_id = fetch_hf_username(hf_token)
    model_dir = "local-qwen-paraphraser"           # 로컬 저장 경로
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"     # 파인튜닝에 사용할 베이스 모델

    # KakaoBrain ParaKQC v1 데이터셋을 GitHub에서 다운로드한다
    # 형식: 탭으로 구분된 (index, label, sentence) — 10개씩 같은 의미 그룹으로 묶여 있음
    print("Downloading KakaoBrain ParaKQC dataset...")
    url = "https://raw.githubusercontent.com/warnikchow/paraKQC/master/data/paraKQC_v1.txt"

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    lines = []
    with urllib.request.urlopen(req) as response:
        for line in response:
            lines.append(line.decode('utf-8').strip())

    # 10개씩 같은 의미 그룹으로 문장을 분류한다
    # group_id = index // 10 으로 동일 의미 그룹을 식별한다
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
    print(f"Loaded {num_sentences} sentences across {len(label_to_sentences)} paraphrase groups.")

    # 같은 그룹 내의 모든 문장 쌍(i≠j)을 패러프레이즈 학습 데이터로 생성한다
    # (sentence_A → sentence_B): A를 입력으로 받아 B를 생성하도록 학습한다
    print("Generating sentence pairs...")
    pairs = []
    for group_id, sentences in label_to_sentences.items():
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    pairs.append({"input": sentences[i], "target": sentences[j]})

    # 재현성을 위해 seed를 고정하고 섞은 뒤, 5000쌍으로 제한한다
    random.seed(42)
    random.shuffle(pairs)
    pairs = pairs[:5000]      # 학습 속도와 메모리를 고려해 5000쌍만 사용
    print(f"Generated {len(pairs)} training pairs.")

    # 4500:500 비율로 학습/검증 세트를 분리한다
    split_idx = 4500
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset   = Dataset.from_list(val_pairs)

    # Qwen 0.5B 토크나이저와 모델을 불러온다
    print(f"Loading tokenizer and model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # float32로 로드 후 fp16 학습 — 정밀도 손실 없이 mixed precision 학습 가능
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # SFT용 시스템 프롬프트: 패러프레이즈 생성 역할을 지정한다
    system_prompt = "You are a question paraphrasing expert. Generate exactly 1 semantic variation of the given question."

    def preprocess_function(examples):
        # Chat Template 기반 SFT 전처리
        # 전체 대화(system + user + assistant)를 하나의 시퀀스로 만들고,
        # 프롬프트 부분(system + user)에는 -100 마스킹을 적용하여
        # 모델이 assistant 응답(target)에 대해서만 손실을 계산하도록 한다
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for inp, tgt in zip(examples["input"], examples["target"]):
            # 전체 대화: system → user(입력 질문) → assistant(목표 패러프레이즈)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: \"{inp}\""},
                {"role": "assistant", "content": tgt}
            ]

            # apply_chat_template: Qwen의 특수 토큰 형식에 맞게 대화를 직렬화한다
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(text, max_length=256, truncation=True)

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 프롬프트(system + user) 부분의 토큰 수를 계산한다
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: \"{inp}\""}
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer(prompt_text)["input_ids"])

            # 프롬프트 토큰에 -100을 적용한다 (-100은 CrossEntropyLoss에서 무시되는 값)
            # 결과: 모델이 assistant 응답 부분에 대해서만 손실을 계산한다
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # 모든 시퀀스를 256 길이로 패딩하거나 잘라낸다
            padding_len = 256 - len(input_ids)
            if padding_len > 0:
                input_ids      = input_ids      + [tokenizer.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                labels         = labels         + [-100] * padding_len  # 패딩 부분도 손실 계산에서 제외
            else:
                input_ids      = input_ids[:256]
                attention_mask = attention_mask[:256]
                labels         = labels[:256]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }

    print("Preprocessing datasets...")
    # 학습/검증 데이터셋 전체에 전처리 함수를 적용한다
    train_tokenized = train_dataset.map(preprocess_function, batched=True, remove_columns=["input", "target"])
    val_tokenized   = val_dataset.map(preprocess_function,   batched=True, remove_columns=["input", "target"])

    # HuggingFace 토큰이 있으면 Hub 업로드를 활성화한다
    should_push = (hf_token is not None and user_id is not None)

    # 학습 하이퍼파라미터를 설정한다
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,                       # SFT에서 흔히 사용하는 학습률
        per_device_train_batch_size=2,            # 6GB VRAM에 맞게 배치 크기를 줄임
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,            # 유효 배치 크기 = 2 × 4 = 8
        fp16=True,                                # mixed precision으로 메모리 절약
        weight_decay=0.01,
        save_total_limit=2,                       # 체크포인트 최대 2개만 보관
        num_train_epochs=1,                       # 0.5B 모델은 1 에폭으로 충분
        logging_steps=50,
        push_to_hub=should_push,
        hub_model_id=f"{user_id}/{model_dir}" if should_push else None,
        hub_token=hf_token if should_push else None,
        report_to="none"
    )

    # Trainer로 학습을 실행한다
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
    )

    print("Training Qwen paraphraser model...")
    trainer.train()  # SFT 학습 시작

    # 학습된 모델과 토크나이저를 로컬에 저장한다
    print("Saving the best model locally...")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}/")

    # 토큰이 유효한 경우 HuggingFace Hub에도 업로드한다
    if should_push:
        print("Pushing to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Success! Model uploaded to: https://huggingface.co/{user_id}/{model_dir}")

if __name__ == "__main__":
    main()
