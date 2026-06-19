# NestJS MCP 서버(port 8000)의 요청을 받아 GPU 기반 AI 추론을 수행하는 Flask 서버
# NestJS는 Python 라이브러리를 직접 호출할 수 없으므로 이 서버가 중간 계층으로 동작한다
# 세 가지 AI 모델을 서버 시작 시 메모리에 올려두고 HTTP 요청마다 추론한다

import os
import json
import re
import torch
from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

app = Flask(__name__)

# --- 환경 변수에서 모델 ID와 서버 포트를 읽어온다 ---
# 기본값이 설정되어 있으므로 .env 없이도 동작하지만, 커스텀 모델을 사용하려면 .env에 지정한다
NLI_MODEL_ID = os.environ.get('HF_STAGE2_MODEL_ID', 'serize/klue-roberta-base-nli-stage2')
PARA_MODEL_ID = os.environ.get('HF_PARA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
PORT = int(os.environ.get('NLI_SERVER_PORT', '8001'))

# NVIDIA GPU가 있으면 cuda, 없으면 cpu를 사용한다
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[nli_server] Using device: {device}")

# --- 모델 1: NLI 분류 모델 로드 ---
# klue/roberta-base를 KLUE-NLI로 파인튜닝한 3-class 분류 모델
# /nli 엔드포인트에서 전제-가설 쌍의 함의 관계를 추론하는 데 사용된다 (Layer 2, 3)
print(f"[nli_server] Loading NLI model: {NLI_MODEL_ID}")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(device)
nli_model.eval()  # 추론 모드로 설정 (드롭아웃 비활성화)
print(f"[nli_server] NLI Model loaded. Labels: {nli_model.config.id2label}")

# --- 모델 2: 패러프레이저 로드 ---
# 우선 로컬에 파인튜닝된 모델(./local-qwen-paraphraser)을 확인한다
# 없으면 베이스 Qwen 0.5B 모델을 폴백으로 사용한다
PARAPHRASE_MODEL_DIR = './local-qwen-paraphraser'
if not os.path.exists(PARAPHRASE_MODEL_DIR):
    # 상대 경로로 찾지 못한 경우 스크립트 파일 기준 절대 경로로 재시도한다
    PARAPHRASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'local-qwen-paraphraser')

if not os.path.exists(PARAPHRASE_MODEL_DIR):
    # 로컬 파인튜닝 모델이 없으면 HuggingFace Hub의 베이스 모델을 사용한다
    PARA_MODEL_ID = os.environ.get('HF_PARA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
    print(f"[nli_server] Local paraphraser not found. Using base model: {PARA_MODEL_ID}")
else:
    PARA_MODEL_ID = PARAPHRASE_MODEL_DIR
    print(f"[nli_server] Loading local paraphraser from: {PARA_MODEL_ID}")

print(f"[nli_server] Loading Paraphrase model: {PARA_MODEL_ID}")
para_tokenizer = AutoTokenizer.from_pretrained(PARA_MODEL_ID)
if device == "cuda":
    # GPU에서는 float16으로 로드하여 메모리를 절약한다
    para_model = AutoModelForCausalLM.from_pretrained(PARA_MODEL_ID, torch_dtype=torch.float16).to(device)
else:
    # CPU에서는 float16보다 지원이 안정적인 bfloat16을 사용한다
    para_model = AutoModelForCausalLM.from_pretrained(PARA_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
para_model.eval()
print("[nli_server] Paraphrase model loaded.")

# --- 모델 3: 답변 생성 모델(Qwen 3B) 로드 ---
# Layer 2 SINdex에서 동일 질문을 여러 번 샘플링하는 데 사용된다
GEN_MODEL_ID = os.environ.get('HF_GEN_MODEL_ID', 'Qwen/Qwen2.5-3B-Instruct')
print(f"[nli_server] Loading general QA generation model: {GEN_MODEL_ID}")

# 생성 파라미터 기본값 (환경 변수로 재정의 가능)
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', '0.3'))   # 낮을수록 결정적 응답
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', '0.95'))              # nucleus sampling 확률 임계값
GEN_MAX_NEW_TOKENS = int(os.environ.get('GEN_MAX_NEW_TOKENS', '128'))
GEN_SYSTEM_PROMPT = os.environ.get(
    'GEN_SYSTEM_PROMPT',
    "You are a precise factual assistant. Answer only based on verified knowledge. "
    "If you are not sure about a fact, say you don't know. "
    "Answer concisely in Korean."
)

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)

if device == "cuda":
    # 4-bit NF4 양자화 설정: 3B 모델을 6GB VRAM에 올리기 위해 필수적으로 적용한다
    # double_quant=True: 양자화 상수 자체도 양자화하여 추가 메모리 절약
    # bnb_4bit_compute_dtype=float16: 연산은 float16으로 수행하여 속도를 유지한다
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',  # NormalFloat4: 정규분포 가중치에 최적화된 양자화 방식
    )
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map='auto',  # 여러 GPU 또는 GPU+CPU에 자동으로 레이어를 분산 배치한다
    )
    print("[nli_server] General QA model loaded on GPU (4-bit quantized).")
else:
    # CPU/Intel GPU 환경: bitsandbytes는 CUDA 전용이므로 양자화 없이 로드한다
    print("[nli_server] Disabling 4-bit quantization on CPU/Intel GPU (no CUDA support).")
    try:
        gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    except Exception as e:
        # bfloat16도 실패하면 기본 dtype(float32)으로 재시도한다
        print(f"[nli_server] Failed to load with bfloat16: {e}. Falling back to default dtype.")
        gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_ID, device_map='auto')
    print("[nli_server] General QA model loaded on CPU/non-CUDA.")

gen_model.eval()
print("[nli_server] General QA model (3B, 4-bit) loaded.")


@app.route('/health')
def health():
    # 서버 상태와 현재 로드된 모델 ID를 반환한다
    # NestJS 서버가 시작 전 이 엔드포인트로 Python 서버 준비 여부를 확인한다
    return jsonify({
        'status': 'ok',
        'nli_model': NLI_MODEL_ID,
        'paraphrase_model': PARA_MODEL_ID
    })


@app.route('/nli', methods=['POST'])
def nli():
    # premise(전제)와 hypothesis(가설) 두 문장이 entailment/neutral/contradiction 관계인지 판별한다
    # Layer 2(SINdex)와 Layer 3(SAC³)에서 원본 응답과 샘플 응답 간의 함의 관계를 계산할 때 호출된다
    data = request.get_json(force=True)
    premise   = data.get('premise', '')
    hypothesis = data.get('hypothesis', '')

    if not premise or not hypothesis:
        return jsonify({'error': 'premise and hypothesis required'}), 400

    # 두 문장을 [CLS] premise [SEP] hypothesis [SEP] 형태로 토크나이즈한다
    inputs = nli_tokenizer(
        premise, hypothesis,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    # 그래디언트 계산 없이 순전파만 수행한다 (추론 전용)
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # logits를 softmax로 확률로 변환한다
    probs = torch.softmax(logits, dim=-1)[0]
    labels = nli_model.config.id2label

    # entailment 레이블에 해당하는 확률 점수를 추출한다
    # 레이블 이름이 모델마다 다를 수 있으므로 이름에 'entail'이 포함된 것을 찾는다
    entailment_score = 0.5  # 찾지 못했을 때의 기본값
    all_scores = {}
    for idx, label in labels.items():
        score = probs[idx].item()
        all_scores[label] = round(score, 4)
        if label.lower() in ('entailment', 'label_0') or 'entail' in label.lower():
            entailment_score = score

    return jsonify({'entailment': entailment_score, 'scores': all_scores})


@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    # 입력 질문을 의미는 같지만 표현이 다른 3개의 변형 질문으로 변환한다
    # Layer 3(SAC³)에서 원본 질문의 패러프레이즈를 생성할 때 호출된다
    data = request.get_json(force=True)
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'question is required'}), 400

    # 패러프레이즈 3개를 JSON 배열 형태로 출력하도록 지시하는 프롬프트
    system_prompt = (
        "You are a question paraphrasing expert. Generate exactly 3 semantic variations of the given question. "
        "They must ask for the exact same factual information but using different phrasing or structure in the same language. "
        "Output ONLY a valid JSON string array of the 3 questions, for example: [\"question 1\", \"question 2\", \"question 3\"]"
    )
    user_prompt = f"Question: \"{question}\""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Qwen의 chat template 형식으로 프롬프트를 직렬화한 후 토크나이즈한다
    text = para_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = para_tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = para_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,   # 약간의 다양성을 허용하여 변형 질문이 서로 달라지게 한다
            do_sample=True
        )

    # 입력 토큰을 제외한 생성된 토큰만 추출하여 디코딩한다
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = para_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    paraphrases = []

    # 1차 시도: JSON 배열 형식으로 파싱한다
    try:
        # 모델이 설명 텍스트를 앞에 붙이는 경우 '[' ~ ']' 사이만 추출한다
        json_match = re.search(r'\[\s*".*?"\s*\]', response, re.DOTALL)
        clean_response = json_match.group(0) if json_match else response
        parsed = json.loads(clean_response)
        if isinstance(parsed, list):
            paraphrases = [str(q).strip() for q in parsed if q]
    except Exception:
        # 2차 시도: JSON 파싱이 실패하면 큰따옴표로 감싸인 문자열을 정규식으로 추출한다
        quotes_match = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)
        if quotes_match and len(quotes_match) >= 3:
            paraphrases = [q.strip() for q in quotes_match]

    # 원본 질문과 동일하거나 빈 문자열인 항목을 제거한다
    paraphrases = [p for p in paraphrases if p and p != question]

    # 3차 시도: 위 방법 모두 실패하면 줄바꿈으로 분리해 번호/기호를 제거한다
    if not paraphrases:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        for line in lines:
            cleaned = re.sub(r'^\d+[\.\-\)]\s*', '', line).strip('"\'  ')
            if cleaned and cleaned != question and cleaned not in paraphrases:
                paraphrases.append(cleaned)

    # 최종 보장: 3개 미만이면 원본 질문을 가공하여 채운다
    while len(paraphrases) < 3:
        idx = len(paraphrases) + 1
        paraphrases.append(f"다른 각도에서의 질문 {idx}: {question}")

    return jsonify({'paraphrases': paraphrases[:3]})


@app.route('/generate', methods=['POST'])
def generate():
    # 질문(prompt)을 받아 Qwen 3B 모델로 짧은 사실 기반 답변을 생성한다
    # Layer 2(SINdex)에서 동일 질문에 대해 여러 번 샘플링하여 응답 일관성을 측정하는 데 사용된다
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    # system_prompt가 요청에 포함되지 않으면 기본 시스템 프롬프트를 사용한다
    system_prompt = data.get('system_prompt', GEN_SYSTEM_PROMPT)

    # 요청에 포함된 파라미터가 있으면 우선 적용하고, 없으면 환경 변수 기본값을 사용한다
    temperature    = float(data.get('temperature', GEN_TEMPERATURE))
    top_p          = float(data.get('top_p', GEN_TOP_P))
    max_new_tokens = int(data.get('max_new_tokens', GEN_MAX_NEW_TOKENS))

    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400

    # 시스템 프롬프트와 사용자 질문을 chat 형식으로 구성한다
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        # Qwen chat template 형식으로 직렬화하고 토크나이즈한다
        text = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = gen_tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.1,  # 0이면 최소값 0.1로 대체
                top_p=top_p,
                do_sample=(temperature > 0)  # temperature가 0이면 greedy decoding 사용
            )

        # 입력 토큰을 제외한 생성 부분만 디코딩한다
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = gen_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"[nli_server] Starting on port {PORT}")
    # threaded=True: 여러 요청을 동시에 처리할 수 있도록 멀티스레드 모드로 실행한다
    app.run(host='0.0.0.0', port=PORT, threaded=True)
