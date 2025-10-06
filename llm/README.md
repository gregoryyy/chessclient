# Finetuning Workflow

(I decided to start this from scratch to be sure to use the most recent approaches. Much has happened in the last years. When I started, HF wasn't that much of an ecosystem, MLC didn't exist, Mistral same...)

Idea: 
- Improve the quality of recommendations by the coach
- Allow LLM to play against Stockfish -- neural vs. symbolic AI
- Test LLM capabilities with AlphaGo-style RL with auto-play
- Show simple round-trip workflow of model improvement

## 1. Prerequisites

The architecture is based on MLC format and WebLLM deployment. However, any finetuning requires an unquantized uncompiled model format, e.g., in HF / PyTorch format.

### Base models

- Assuming we use HF: https://huggingface.co/
- HF base models:
  - Example used: Llama 3.2 1B Instruct:
    - Model page: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
    - Need to register/apply for license in HF
  - Typical case: Model access via cache, see `/hfmodel.py`
      - Loads to `~/.cache/huggingface/hub/`
      - List models: `huggingface-cli scan-cache`
  - For explicit download needed (e.g., for low-level hacks):
    ```python
    from huggingface_hub import snapshot_download
    model = "meta-llama/Llama-3.2-1B-Instruct"
    snapshot_download(model, local_dir="./models/" + model)
    model = AutoModelForCausalLM.from_pretrained("./models/" + model)
    ```
    or
    ```bash
    huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir Llama-3.2-1B-Instruct
    ```
- Formats for `lora.py`:
  - Safetensors `.safetensors` -- fully supported; no pickles; faster load.
  - PyTorch `.bin` -- works, but slower; converted internally to TVM weights.
  - Merged LoRA model dir (FP16 HF format `config.json`, `model.safetensors` or `.bin`, tokenizer files).
  - HF Hub repo -- `mlc_llm build --model meta-llama/Llama-2-7b-chat-hf` works; downloads automatically.
  - Not supported:
    - Quantized (bitsandbytes / GGUF / GPTQ / AWQ) -- MLC quantizes itself
    - ONNX / TorchScript / TFLite -- MLC needs original HF checkpoints, not converted graphs.
  - Convert using MLC toolchain (see below) after finetuning
- Model format supported by WebLLM (after/without finetuning):
  - Target format for all models to be used in the Checkle system.
    - Root URL: - https://huggingface.co/mlc-ai
  - Example used: Llama 3.2 1B Instruct (same as above)
    - Model page: https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC
    - Download with Git Large File Storage: 
      ```bash
      git lfs install
      git clone https://huggingface.co/mlc-ai/Llama-3.2-1B-q4f16_1-MLC
      ```
    - Run with `mlc_llm {chat|serve} HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_0-MLC`
    - Python: see `server_mlc.py`

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

Execute `lora.py`

```bash
# create an isolated environment for LoRA finetuning
python3.11 -m venv .venv
. .venv/bin/activate

# install the trainer stack (Transformers, PEFT, TRL, datasets, etc.)
pip install pip-tools
# using minimal requirements.in for deps
pip-compile
pip install -r requirements.txt

# run the LoRA finetuning script against the configured base model
python lora.py
```
```bash
   python lora.py --base models/Meta-Llama-3-8B-Instruct \
     --data ./datasets/sft.jsonl --out ./artifacts/llama3-demo \
     --qlora true --epochs 1 --mlc-target metal --mlc-quant q4f16_1

   python lora.py --base mistralai/Mistral-7B-Instruct-v0.3 \
     --data ./datasets/sft.jsonl --out ./artifacts/mistral-demo \
     --distill true --distill-max-samples 200 --mlc-target cuda --mlc-quant q8f16
```

### MLC Export

- Convert finetuned LLM to the MLC format for WebLLM usage (included in lora.py above)
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

## 3. Validation

### Server

Use the server scripts or command lines:

- HF/Safetensors: `server.py` (adjust before)
- MLC: `server_mlc.py` (adjust before)
- Shell: `mlc_llm {serve|chat} <model>`

### Browser models

- Run a quick regression: 
  - Push a few held-out prompts through both the original and fine-tuned models to verify behavior.
  - In the browser, watch console/logs to ensure the custom bundle loads (progress reaches 100%) and responses look correct.
