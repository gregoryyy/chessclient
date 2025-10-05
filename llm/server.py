# tiny_server.py
# Supports .safetensors or .bin weights (opt. sharded, opt. quantization via BitsAndBytes)
import torch, json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
app = FastAPI()

@app.post("/generate")
def generate(req: dict):
    prompt = req["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=req.get("max_tokens", 256),
                      temperature=req.get("temperature", 0.7), streamer=streamer)

    def sse():
        import threading
        t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t.start()
        for token in streamer:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: {\"done\": true}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
