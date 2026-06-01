#!/bin/sh
set -e

MODEL_DIR="./local-qwen-paraphraser"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "[entrypoint] Downloading paraphraser model from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${HF_PARA_MODEL_ID:-serize/local-qwen-paraphraser}',
    local_dir='$MODEL_DIR',
    token='${HF_API_KEY:-}'
)
print('[entrypoint] Model download complete.')
"
else
    echo "[entrypoint] Model already present at $MODEL_DIR"
fi

exec python nli_server.py
