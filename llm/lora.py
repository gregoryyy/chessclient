# lora_quant_mlc.py
# Usage examples:
#   python lora_quant_mlc.py --base meta-llama/Meta-Llama-3-8B-Instruct \
#     --data ./datasets/sft.jsonl --out ./artifacts/llama3-demo \
#     --qlora true --epochs 1 --mlc-target metal --mlc-quant q4f16_1
#
#   python lora_quant_mlc.py --base mistralai/Mistral-7B-Instruct-v0.3 \
#     --data ./datasets/sft.jsonl --out ./artifacts/mistral-demo \
#     --distill true --distill-max-samples 200 --mlc-target cuda --mlc-quant q8f16

import os, json, subprocess, tempfile
from pathlib import Path
from typing import List
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig, TextIteratorStreamer
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType, AutoPeftModelForCausalLM
import torch

def load_jsonl_dataset(path: str, text_field: str = "text"):
    # Expect one JSON object per line with {"text": "..."}
    # Map to HF datasets under the same field.
    ds = load_dataset("json", data_files=path, split="train")
    assert text_field in ds.column_names, f"Dataset must contain field '{text_field}'"
    return ds

def build_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def build_base_model(base: str, qlora: bool, attn_impl: str = "sdpa"):
    if qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return AutoModelForCausalLM.from_pretrained(
            base,
            quantization_config=bnb,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    else:
        # Full-precision (BF16 preferred), handled by Trainer args
        return None  # SFTTrainer will load from model_name_or_path

def train_lora(
    base: str, ds, out_dir: Path,
    max_seq_len: int, packing: bool,
    epochs: int, bs: int, grad_accum: int, lr: float,
    bf16: bool, attn_impl: str, qlora: bool,
    lora_r: int, lora_alpha: int, lora_dropout: float,
    target_modules: List[str]
):
    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", target_modules=target_modules, task_type=TaskType.CAUSAL_LM
    )
    tok = build_tokenizer(base)
    model = build_base_model(base, qlora, attn_impl)

    args = TrainingArguments(
        output_dir=str(out_dir / "adapter"),
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        num_train_epochs=epochs, logging_steps=10,
        evaluation_strategy="no",
        save_strategy="steps", save_steps=200, save_total_limit=2,
        gradient_checkpointing=True, bf16=bf16, report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        model_name_or_path=None if model is not None else base,
        tokenizer=tok,
        peft_config=peft_cfg,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=packing,
        args=args,
        model_init_kwargs=dict(attn_implementation=attn_impl),
    )
    trainer.train()
    (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    return out_dir / "adapter"

@torch.no_grad()
def self_distill_pseudolabels(teacher_path: Path, data_path: str, out_jsonl: Path, max_samples: int = 200, max_new_tokens: int = 256):
    """
    Lightweight self-distillation:
    - Load merged teacher
    - Generate teacher answers appended to the input text as targets
    - Write a new JSONL with combined text for a second quick SFT pass
    Assumes dataset lines are prompts under {"text": "..."}.
    """
    tok = AutoTokenizer.from_pretrained(teacher_path, use_fast=True)
    model = AutoPeftModelForCausalLM.from_pretrained(teacher_path, device_map="auto")
    model.eval()

    ds = load_dataset("json", data_files=data_path, split="train")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(ds), max_samples)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(n):
            prompt = ds[i]["text"]
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
            gen_kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, streamer=streamer)
            # Generate in a background thread for streaming (not strictly needed)
            import threading
            t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            t.start()
            out = ""
            for chunk in streamer:
                out += chunk
            # Combine as a supervised target sample
            combined = f"{prompt}\n\n### Answer:\n{out.strip()}"
            f.write(json.dumps({"text": combined}, ensure_ascii=False) + "\n")

def merge_adapter_to_fp16(adapter_path: Path, merged_out: Path):
    merged = AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), device_map="auto").merge_and_unload()
    merged_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_path)).save_pretrained(merged_out)
    return merged_out

def compile_to_mlc(hf_dir: Path, mlc_out: Path, target: str, quant: str):
    """
    Calls the MLC compiler. Common targets: cuda | metal | webgpu
    Common quant: q4f16_1 | q4f32_1 | q8f16
    """
    mlc_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "mlc_llm.build",
        "--model", str(hf_dir),
        "--artifact-path", str(mlc_out),
        "--target", target,
        "--quantization", quant
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser("LoRA → (optional QLoRA) → (optional self-distill) → MLC export")
    p.add_argument("--base", required=True, help="HF model id or local dir")
    p.add_argument("--data", required=True, help="JSONL path with {'text': ...}")
    p.add_argument("--out", required=True, help="Output root directory")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--packing", type=bool, default=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bf16", type=bool, default=True)
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa","eager"])
    p.add_argument("--qlora", type=bool, default=False)

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", nargs="+", default=["q_proj","k_proj","v_proj","o_proj"])

    p.add_argument("--distill", type=bool, default=False, help="Run a tiny self-distillation pass")
    p.add_argument("--distill-max-samples", type=int, default=200)
    p.add_argument("--distill-newdata", default=None, help="Optional path to write distilled JSONL")

    p.add_argument("--mlc-target", default="metal", choices=["metal","cuda","webgpu"])
    p.add_argument("--mlc-quant", default="q4f16_1", help="MLC quantization preset")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load dataset (validated by trainer later)
    ds = load_jsonl_dataset(args.data)

    # 2) LoRA / QLoRA fine-tune → adapter/
    adapter_dir = train_lora(
        base=args.base, ds=ds, out_dir=out,
        max_seq_len=args.max_seq_len, packing=args.packing,
        epochs=args.epochs, bs=args.bs, grad_accum=args.grad_accum, lr=args.lr,
        bf16=args.bf16, attn_impl=args.attn_impl, qlora=args.qlora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target
    )

    # 3) Merge adapter → FP16 HF dir (needed for MLC build)
    merged_dir = merge_adapter_to_fp16(adapter_dir, out / "merged-fp16")
    print(f"Merged model at: {merged_dir}")

    # 4) (Optional) Self-distillation pass (generate pseudo-labels, quick SFT)
    if args.distill:
        distilled_jsonl = Path(args.distill_newdata) if args.distill_newdata else (out / "distilled.jsonl")
        print("Generating pseudo-labels for self-distillation…")
        self_distill_pseudolabels(teacher_path=merged_dir, data_path=args.data,
                                  out_jsonl=distilled_jsonl, max_samples=args.distill_max_samples)
        # quick extra epoch on distilled data
        ds2 = load_jsonl_dataset(str(distilled_jsonl))
        # Re-run SFT briefly using merged as base (no LoRA now):
        tok = build_tokenizer(str(merged_dir))
        model = AutoModelForCausalLM.from_pretrained(str(merged_dir), device_map="auto", attn_implementation=args.attn_impl)
        args2 = TrainingArguments(
            output_dir=str(out / "distilled"),
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=5e-5, lr_scheduler_type="cosine", warmup_ratio=0.03,
            num_train_epochs=1, logging_steps=10, report_to=["none"],
            gradient_checkpointing=True, bf16=True
        )
        trainer2 = SFTTrainer(
            model=model, tokenizer=tok, model_name_or_path=None,
            dataset_text_field="text", max_seq_length=args.max_seq_len, packing=True, args=args2
        )
        trainer2.train()
        trainer2.model.save_pretrained(out / "distilled" / "merged-fp16")
        tok.save_pretrained(out / "distilled" / "merged-fp16")
        merged_dir = out / "distilled" / "merged-fp16"
        print(f"Distilled merged model at: {merged_dir}")

    # 5) Export to MLC (quantization chosen here)
    mlc_out = out / f"mlc-{args.mlc_target}-{args.mlc_quant}"
    compile_to_mlc(merged_dir, mlc_out, target=args.mlc_target, quant=args.mlc_quant)
    print(f"MLC package ready at: {mlc_out}")

if __name__ == "__main__":
    main()
