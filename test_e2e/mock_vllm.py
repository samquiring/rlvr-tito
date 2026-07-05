"""Mock vLLM OpenAI server backed by a local HF model.

Implements the three behaviors rlvr-tito depends on, with the same JSON
shapes as vLLM:

  POST /v1/chat/completions   -> message.content, prompt_token_ids (top level),
                                 choices[0].token_ids,
                                 choices[0].logprobs.content[{token, logprob}]
  POST /v1/load_lora_adapter  -> loads a PEFT adapter at runtime
  routing by request "model"  -> adapter name selects the active adapter

Notes: single-worker CPU server; behavior logprobs are the sampling-
distribution logprobs (test runs temperature=1.0 so they equal raw model
logprobs, matching what the trainer recomputes).
"""

import os
import threading

import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", "./tiny-qwen3")

app = FastAPI(title="mock-vllm")
_lock = threading.Lock()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
base.eval()
state = {"model": base, "adapters": set(), "active": None}


@app.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: Request):
    body = await request.json()
    name, path = body["lora_name"], body["lora_path"]
    with _lock:
        from peft import PeftModel
        if not state["adapters"]:
            state["model"] = PeftModel.from_pretrained(
                base, path, adapter_name=name, is_trainable=False)
        else:
            state["model"].load_adapter(path, adapter_name=name)
        state["adapters"].add(name)
        state["active"] = name
        state["model"].set_adapter(name)
        state["model"].eval()
    return {"ok": True, "active": name}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body["messages"]
    temperature = float(body.get("temperature", 1.0))
    max_tokens = int(body.get("max_tokens", 32))
    requested = body.get("model", "")

    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids])

    with _lock:
        model = state["model"]
        if requested in state["adapters"] and requested != state["active"]:
            model.set_adapter(requested)
            state["active"] = requested
        with torch.no_grad():
            out = model.generate(
                input_ids,
                do_sample=True, temperature=temperature,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
            )

    resp_ids = out.sequences[0, input_ids.shape[1]:].tolist()
    # logprob of each sampled token under the (temperature-processed)
    # sampling distribution, matching vLLM's sampled-token logprobs
    logprobs = []
    for step, tok_id in enumerate(resp_ids):
        lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tok_id]
        logprobs.append(float(lp))

    text = tokenizer.decode(
        [t for t in resp_ids if t != tokenizer.eos_token_id])

    return {
        "id": "mock",
        "object": "chat.completion",
        "model": requested or "base",
        "prompt_token_ids": list(prompt_ids),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "token_ids": resp_ids,
            "logprobs": {"content": [
                {"token": tokenizer.decode([t]), "logprob": lp}
                for t, lp in zip(resp_ids, logprobs)
            ]},
            "finish_reason": "stop",
        }],
    }


@app.get("/health")
async def health():
    return {"ok": True, "active": state["active"]}
