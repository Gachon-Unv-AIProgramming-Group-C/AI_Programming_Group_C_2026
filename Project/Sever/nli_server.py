import os
import json
import re
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig

app = Flask(__name__)

# --- Configuration ---
NLI_MODEL_ID = os.environ.get('HF_STAGE2_MODEL_ID', 'serize/klue-roberta-base-nli-stage2')
PARA_MODEL_ID = os.environ.get('HF_PARA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
PORT = int(os.environ.get('NLI_SERVER_PORT', '8001'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[nli_server] Using device: {device}")

# --- Load Models ---
print(f"[nli_server] Loading NLI model: {NLI_MODEL_ID}")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(device)
nli_model.eval()
print(f"[nli_server] NLI Model loaded. Labels: {nli_model.config.id2label}")

# Check if local trained paraphraser exists, otherwise use base/public model as fallback
PARAPHRASE_MODEL_DIR = './local-qwen-paraphraser'
if not os.path.exists(PARAPHRASE_MODEL_DIR):
    PARAPHRASE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'local-qwen-paraphraser')

if not os.path.exists(PARAPHRASE_MODEL_DIR):
    PARA_MODEL_ID = os.environ.get('HF_PARA_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
    print(f"[nli_server] Local paraphraser not found. Using base model: {PARA_MODEL_ID}")
else:
    PARA_MODEL_ID = PARAPHRASE_MODEL_DIR
    print(f"[nli_server] Loading local paraphraser from: {PARA_MODEL_ID}")

print(f"[nli_server] Loading Paraphrase model: {PARA_MODEL_ID}")
para_tokenizer = AutoTokenizer.from_pretrained(PARA_MODEL_ID)
if device == "cuda":
    para_model = AutoModelForCausalLM.from_pretrained(PARA_MODEL_ID, torch_dtype=torch.float16).to(device)
else:
    # Use bfloat16 on CPU as it is better supported and faster than float16
    para_model = AutoModelForCausalLM.from_pretrained(PARA_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
para_model.eval()
print("[nli_server] Paraphrase model loaded.")

# Load general generation model (Qwen2.5-3B-Instruct, 4-bit quantized for RTX 2060 6GB)
GEN_MODEL_ID = os.environ.get('HF_GEN_MODEL_ID', 'Qwen/Qwen2.5-3B-Instruct')
print(f"[nli_server] Loading general QA generation model: {GEN_MODEL_ID}")

# Generation parameter defaults (can be overridden via env vars)
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', '0.3'))
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', '0.95'))
GEN_MAX_NEW_TOKENS = int(os.environ.get('GEN_MAX_NEW_TOKENS', '128'))
GEN_SYSTEM_PROMPT = os.environ.get(
    'GEN_SYSTEM_PROMPT',
    "You are a precise factual assistant. Answer only based on verified knowledge. "
    "If you are not sure about a fact, say you don't know. "
    "Answer concisely in Korean."
)

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)

if device == "cuda":
    # 4-bit quantization config to fit 7B model within 6GB VRAM (only for NVIDIA GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map='auto',
    )
    print("[nli_server] General QA model loaded on GPU (4-bit quantized).")
else:
    # On CPU/Intel GPU (no CUDA), bitsandbytes is not supported and will hang.
    # Load model in bfloat16 or float32 directly without quantization.
    print("[nli_server] Disabling 4-bit quantization on CPU/Intel GPU (no CUDA support).")
    try:
        gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    except Exception as e:
        print(f"[nli_server] Failed to load with bfloat16: {e}. Falling back to default dtype.")
        gen_model = AutoModelForCausalLM.from_pretrained(
            GEN_MODEL_ID,
            device_map='auto',
        )
    print("[nli_server] General QA model loaded on CPU/non-CUDA.")

gen_model.eval()
print("[nli_server] General QA model (7B, 4-bit) loaded.")

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'nli_model': NLI_MODEL_ID,
        'paraphrase_model': PARA_MODEL_ID
    })

@app.route('/nli', methods=['POST'])
def nli():
    data = request.get_json(force=True)
    premise = data.get('premise', '')
    hypothesis = data.get('hypothesis', '')

    if not premise or not hypothesis:
        return jsonify({'error': 'premise and hypothesis required'}), 400

    inputs = nli_tokenizer(
        premise, hypothesis,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    labels = nli_model.config.id2label

    # Find entailment score (label_0 or contains 'entail')
    entailment_score = 0.5
    all_scores = {}
    for idx, label in labels.items():
        score = probs[idx].item()
        all_scores[label] = round(score, 4)
        if label.lower() in ('entailment', 'label_0') or 'entail' in label.lower():
            entailment_score = score

    return jsonify({'entailment': entailment_score, 'scores': all_scores})

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.get_json(force=True)
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'question is required'}), 400

    # Instruct prompt for Qwen to generate paraphrases
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
    
    text = para_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = para_tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = para_model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True
        )

    # Extract generated output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = para_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    paraphrases = []
    
    # Try parsing response as JSON array
    try:
        # Find first '[' and last ']' to extract JSON if there's conversational wrap
        json_match = re.search(r'\[\s*".*?"\s*\]', response, re.DOTALL)
        clean_response = json_match.group(0) if json_match else response
        parsed = json.loads(clean_response)
        if isinstance(parsed, list):
            paraphrases = [str(q).strip() for q in parsed if q]
    except Exception:
        # Fallback regex parsing if JSON format is slightly malformed
        # e.g., finding all double-quoted strings
        quotes_match = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)
        if quotes_match and len(quotes_match) >= 3:
            paraphrases = [q.strip() for q in quotes_match]

    # Clean duplicates and original question
    paraphrases = [p for p in paraphrases if p and p != question]

    # Fallback to line splits if empty
    if not paraphrases:
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        for line in lines:
            cleaned = re.sub(r'^\d+[\.\-\)]\s*', '', line).strip('"\' ')
            if cleaned and cleaned != question and cleaned not in paraphrases:
                paraphrases.append(cleaned)

    # Fallback to make sure we have exactly 3 variations
    while len(paraphrases) < 3:
        idx = len(paraphrases) + 1
        paraphrases.append(f"다른 각도에서의 질문 {idx}: {question}")

    return jsonify({'paraphrases': paraphrases[:3]})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    # Use provided or default system prompt
    system_prompt = data.get('system_prompt', GEN_SYSTEM_PROMPT)

    # Generation parameters with defaults from env vars
    temperature = float(data.get('temperature', GEN_TEMPERATURE))
    top_p = float(data.get('top_p', GEN_TOP_P))
    max_new_tokens = int(data.get('max_new_tokens', GEN_MAX_NEW_TOKENS))

    if not prompt:
        return jsonify({'error': 'prompt is required'}), 400

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        text = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = gen_tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.1,
                top_p=top_p,
                do_sample=(temperature > 0)
            )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = gen_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return jsonify({'text': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"[nli_server] Starting on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
