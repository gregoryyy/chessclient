from mlc_llm import MLCEngine

engine = MLCEngine(model="Llama-3-8B-q4f16_1", device="cuda")  # or "cpu"
out = engine.chat_completion(["What is the capital of France?"])
print(out)

for chunk in engine.chat_stream(["Explain transformers in one line."]):
    print(chunk, end="", flush=True)
