import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Downloading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Tokenizer loaded successfully.")

print(f"Downloading model for {model_id} in float16...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully.")
print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
