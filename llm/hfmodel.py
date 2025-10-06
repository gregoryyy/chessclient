from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-3-270m"   # or "google/gemma-3-270m-it"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
inputs = tok("Explain k-means in one sentence.", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
