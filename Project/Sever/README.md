# Hallucination Detector

A hallucination detection server for LLM responses, exposed as an MCP (Model Context Protocol) tool. Built on the **"Verify when Uncertain"** paper (arXiv:2502.15845), extended into a 3-layer cascade architecture.

## How it works

Detection runs in up to three layers, stopping early when confidence is high enough:

```
Layer 1 — LSC (Lowest Span Confidence)
  If logprobs are provided, low-confidence tokens trigger further checks.

Layer 2 — SINdex (Semantic Inconsistency Index)
  Samples multiple responses to the same question, clusters them, and
  measures how scattered they are. High scatter = likely hallucination.
  (Uses dynamic language-aware Jaccard thresholds: 0.90 for Korean, 0.65 for English/others)

Layer 3 — SAC³ (Cross-model Consistency)
  Paraphrases the original question using our fine-tuned Qwen paraphraser,
  asks an independent verifier, and checks if the answers agree.
  Our fine-tuned KLUE-RoBERTa NLI model scores the entailment.
```

Verdict: `HALLUCINATION` | `NO_HALLUCINATION` | `UNCERTAIN` (manual review recommended)

## Models & Training Types

Our system combines open-source foundation models with custom-trained and fine-tuned models:

### 1. Custom & Fine-Tuned Models (Directly Built/Trained)
*   **`serize/klue-roberta-base-nli-stage2`** (NLI Classifier):
    *   **Role**: Serves as the ultimate semantic consistency judge (Layer 3).
    *   **Training**: Fully fine-tuned by us on the KLUE NLI dataset and uploaded to our Hugging Face repository.
    *   **Link**: [Hugging Face Repository](https://huggingface.co/serize/klue-roberta-base-nli-stage2)
*   **`serize/local-qwen-paraphraser`** (Question Paraphraser):
    *   **Role**: Generates high-quality paraphrased questions to test LLM consistency (Layer 3).
    *   **Training**: Fine-tuned by us using `Qwen2.5-0.5B-Instruct` as the base model and training it on the ParaKQC Korean paraphrase dataset.
    *   **Link**: [Hugging Face Repository](https://huggingface.co/serize/local-qwen-paraphraser)

### 2. External Base Models (Utilized)
*   **`Qwen/Qwen2.5-3B-Instruct`**: Used as the default generator to sample multiple responses in Layer 2 (SINdex) when running queries.

## Project structure

```
.
├── src/
│   └── mcp/
│       ├── mcp.service.ts      # 3-layer cascade algorithm
│       ├── mcp.controller.ts   # POST /mcp HTTP endpoint
│       ├── mcp.module.ts
│       └── mcp.types.ts        # Input/output type definitions
├── nli_server.py               # Python inference server (NLI + paraphraser)
├── train_paraphraser.py        # Local training script for the paraphraser
├── train_paraphraser_colab.ipynb  # Colab notebook version
├── train_stage2_nli.py         # NLI fine-tuning script
├── Dockerfile                  # NestJS server
├── Dockerfile.python           # Python NLI server
├── docker-compose.yml          # Runs both services together
└── requirements-nli.txt        # Python dependencies
```

## Quick start

### Option 1: Docker Compose (recommended)

Requires Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

```bash
# 1. Clone and enter the project
git clone https://github.com/your-org/hallucination-detector
cd hallucination-detector

# 2. Copy env file and fill in your keys
cp .env.example .env

# 3. Start
docker-compose up --build
```

The Python NLI server downloads the paraphraser model from HuggingFace on first start and caches it in a Docker volume. Subsequent starts reuse the cached model.

Services will be available at:
- NestJS MCP server: `http://localhost:8000/mcp`
- Python NLI server: `http://localhost:8001/health`

### Option 2: Local (without Docker)

**Requirements:** Node.js 20+, Python 3.11+, PyTorch (CPU or CUDA)

```bash
# Terminal 1: Start the Python NLI server
pip install -r requirements-nli.txt torch --index-url https://download.pytorch.org/whl/cpu
python -c "
from huggingface_hub import snapshot_download
snapshot_download('serize/local-qwen-paraphraser', local_dir='local-qwen-paraphraser')
"
python nli_server.py

# Terminal 2: Start the NestJS server
npm install
cp .env.example .env
npm run start:dev
```

### Option 3: stdio (Claude Code MCP)

Register the server as a stdio MCP tool in Claude Code:

```bash
npm run build

claude mcp add --transport stdio hallucination-detector \
  node /absolute/path/to/dist/main.js -- --stdio
```

Or add it manually to `~/.claude.json` under `mcpServers`:

```json
{
  "hallucination-detector": {
    "type": "stdio",
    "command": "node",
    "args": ["/absolute/path/to/dist/main.js", "--stdio"],
    "env": {
      "HF_API_KEY": "hf_...",
      "NLI_SERVER_URL": "http://127.0.0.1:8001"
    }
  }
}
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | HTTP server port |
| `OPENAI_API_KEY` | - | OpenAI key (used as LLM verifier) |
| `ANTHROPIC_API_KEY` | - | Anthropic key (used as LLM verifier) |
| `HF_API_KEY` | - | Hugging Face token |
| `HF_MODEL_ID` | `jhgan/ko-sroberta-nli` | Stage 1 NLI model |
| `HF_STAGE2_MODEL_ID` | `serize/klue-roberta-base-nli-stage2` | Stage 2 NLI model |
| `NLI_SERVER_URL` | `http://127.0.0.1:8001` | Python NLI server URL |
| `NLI_SERVER_PORT` | `8001` | Python NLI server port (local only) |
| `LOG_LEVEL` | `info` | Log level |

At least one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` is needed for Layer 2/3 to run LLM verification. If neither is set but the server is running as a Claude Code stdio MCP, it will use client sampling instead.

## Tool API

### `check_hallucination`

**Input**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | yes | The question asked to the LLM |
| `response` | string | yes | The LLM response to check |
| `context` | string | no | Optional source context |
| `m` | integer | no | Number of samples per layer (default: 5) |
| `t1` | float | no | Layer 1 lower threshold (default: 0.3) |
| `tStar` | float | no | Layer 1 upper threshold (default: 0.7) |
| `s2_threshold` | float | no | Layer 2 inconsistency threshold (default: 0.4) |
| `t2` | float | no | Final decision threshold (default: 0.5) |
| `useHuggingFaceNli` | boolean | no | Force HF NLI API (default: false) |

**Output**

```json
{
  "verdict": "HALLUCINATION",
  "is_hallucination": true,
  "confidence": 0.82,
  "reason": "[Layer 3 Detected] Combined risk score 0.82 ...",
  "details": {
    "layersRun": [1, 2, 3],
    "mode": "llm",
    "score1": 0.45,
    "score2": 0.71,
    "score3": 0.81,
    "final_score": 0.72
  }
}
```

When `verdict` is `UNCERTAIN`, the server asks the connected LLM to prompt the user for manual review.

**Example**

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "check_hallucination",
      "arguments": {
        "question": "When did the Hualien earthquake occur?",
        "response": "The magnitude 7.2 earthquake in Hualien, Taiwan occurred on April 2, 2024."
      }
    }
  }'
```

## Training

### NLI model

```bash
python train_stage2_nli.py
```

Fine-tunes `klue/roberta-base` on KLUE NLI data for 3-class entailment classification.

### Paraphraser

**Local:**
```bash
python train_paraphraser.py
```

**Google Colab (recommended — free T4 GPU):**

Open `train_paraphraser_colab.ipynb` in Colab. Training takes about 10 minutes on a T4. The trained model is automatically pushed to your HuggingFace Hub after training.

## Reference

> **Verify when Uncertain: Beyond Self-Consistency in Black Box Hallucination Detection**  
> arXiv:2502.15845, February 2025
