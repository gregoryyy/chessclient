# Finetuning Workflow

Idea: 
- Improve the quality of recommendations by the coach
- Allow LLM to play against Stockfish -- neural vs. symbolic AI
- Test LLM capabilities with AlphaGo-style RL with auto-play
- Show simple round-trip workflow of model improvement

## 1. Prerequisites

The architecture is based on MLC format and WebLLM deployment. However, any finetuning requires an unquantized uncompiled model format, e.g., in HF / PyTorch format.

### Base models

- HF base models:
  - https://huggingface.co/
  - Finetune
  - Convert using MLC toolchain (see below)
- Base model supported by WebLLM (no finetuning):
  - https://huggingface.co/mlc-ai
    - Examples in `/public/models/`
  - Target for all models to be used in the Checkle system.

### Static training data

- Instruction/response pairs:
  - Clean and tokenize (using the same tokenizer as the base model: AutoTokenizer.from_pretrained).
- Sources: 
  - https://github.com/mlabonne/llm-datasets
  - https://medium.com/@valentin.urena/how-to-fine-tune-a-large-language-model-to-play-chess-6da5ee5ab986
    - incl. generating a chess dataset

### Dynamic training

- LLM finetuning using a "dynamic" AlphaGo-style approach:
   - Use a dataset of high-quality chess games (see "static" training data) and 
   - Implement a reinforcement learning (RL) loop
 - LLM plays against itself and learns from its performance, 
   - Evaluating moves and
   - Iteratively improving its ability to select winning moves.

### MLC toolchain

Note: MLC LLM is a machine learning compiler and high-performance deployment engine for large language models, to enable everyone to develop, optimize, and deploy AI models natively on arbitrary platforms, incl. WASM. 

- MLC commands to use generic models:
  - `mlc_llm convert`
  - `mlc_llm quantize`
  - `mlc_llm build-model`  /  `mlc_llm build-web`
- Install for Metal environment:
  - `python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu`
  - see: https://llm.mlc.ai/docs/install/mlc_llm.html

## 2. Finetuning

- Finetuning LLMs with adapters involves freezing the original model weights and training small, lightweight modules inserted into the LLM's architecture. This method, known as Parameter-Efficient Fine-Tuning (PEFT), allows for efficient adaptation to new tasks with significantly fewer trainable parameters, lower memory usage, and faster training times compared to full fine-tuning.
- Adapters provide a modular way to customize a single pre-trained LLM for multiple domains or tasks, making it a cost-effective and flexible solution for deploying specialized LLMs, without without retraining the entire model and risking "catastrophic forgetting". 
- LoRA (Low-Rank Adaptation) is a specific, highly efficient technique for creating adapters by freezing the original LLM weights and training small, low-rank matrices to represent the necessary changes, significantly reducing computation, memory, and storage requirements.



### LoRA

```bash
# create an isolated environment for LoRA finetuning
python3.11 -m venv .venv
source .venv/bin/activate

# install the trainer stack (Transformers, PEFT, TRL, datasets, etc.)
pip install -r requirements.txt

# run the LoRA finetuning script against the configured base model
python lora.py
```

### MLC Export

- Convert finetuned LLM to the MLC format for WebLLM usage
- Pick a quantization that balances quality/perf (WebLLM bundles often use q4f16_1).

```
python -m mlc_llm.convert \
  --model ./llama3-finetune \
  --quantization q4f16_1 \
  --arch llm_chat \
  --target web \
  --output ./artifacts/my-llama3-q4f16_1
```

- Package the generated mlc-chat-config.json, tokenizer files, wasm runtime, and shard binaries. 
- Host the files under your app’s static assets `/public/models` and point CreateMLCEngine at the new bundle (`model_list: [{model_id: "my-llama3", model_url: "/models/my-llama3"}]`).

### Validation

- Run a quick regression: 
- Push a few held-out prompts through both the original and fine-tuned models to verify behavior.
- In the browser, watch console/logs to ensure the custom bundle loads (progress reaches 100%) and responses look correct.


