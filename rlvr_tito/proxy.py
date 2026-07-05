"""TITO proxy: an OpenAI-compatible endpoint that any agent harness can call,
which transparently captures exact token ids + logprobs for RL training.

The agent's contract is deliberately tiny:

  1. POST /trajectories                {"task_id": "..."} -> {"traj_id": ...}
  2. POST /v1/chat/completions        as usual, plus header X-Trajectory-ID
  3. POST /trajectories/{id}/reward   {"reward": 1.0} when the episode ends

Everything else (return_token_ids, logprobs capture, adapter routing, GRPO
grouping) happens here. The agent never sees a token id, never tokenizes, and
can be tau2's gym loop, your own harness, or anything speaking OpenAI chat.

Requires vLLM launched like:

  VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 vllm serve Qwen/Qwen3.5-9B-Instruct \
      --enable-lora --max-lora-rank 32 --port 8000

vLLM's OpenAI endpoints return exact token ids for prompt and generation when
"return_token_ids": true is passed (vLLM >= 0.11). We also request logprobs so
behavior logprobs ride along with every transition.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

from .store import GroupStore, Transition

log = logging.getLogger("rlvr_tito.proxy")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
TRAIN = os.environ.get("TRAIN", "0") == "1"

app = FastAPI(title="rlvr-tito proxy")
store = GroupStore(group_size=GROUP_SIZE)
state = {"adapter": None, "adapter_version": 0}
_client = httpx.AsyncClient(base_url=VLLM_URL, timeout=600)


# --------------------------------------------------------------------------
# Trajectory lifecycle
# --------------------------------------------------------------------------

@app.post("/trajectories")
async def start_trajectory(body: dict):
    task_id = body.get("task_id")
    if not task_id:
        raise HTTPException(400, "task_id required")
    return {"traj_id": store.start_trajectory(task_id)}


@app.post("/trajectories/{traj_id}/reward")
async def complete_trajectory(traj_id: str, body: dict):
    if "reward" not in body:
        raise HTTPException(400, "reward required")
    try:
        store.complete(traj_id, float(body["reward"]))
    except KeyError:
        raise HTTPException(404, f"unknown trajectory {traj_id}")
    return {"ok": True}


@app.get("/stats")
async def stats():
    return {**store.stats, "adapter": state["adapter"]}


# --------------------------------------------------------------------------
# OpenAI-compatible passthrough with token capture
# --------------------------------------------------------------------------

def _extract_tokens(data: dict) -> tuple[list[int], list[int], list[float]]:
    """Pull prompt ids, response ids, and behavior logprobs out of a vLLM
    chat.completions response. Fails loudly on any misalignment: silently
    dropping or padding here is exactly how gradients get corrupted."""
    choice = data["choices"][0]

    prompt_ids = data.get("prompt_token_ids") or choice.get("prompt_token_ids")
    resp_ids = choice.get("token_ids") or data.get("token_ids")
    if prompt_ids is None or resp_ids is None:
        raise HTTPException(
            502,
            "vLLM response lacks token ids. Ensure vLLM >= 0.11 and that "
            "return_token_ids is supported on this endpoint.",
        )

    lp_content = (choice.get("logprobs") or {}).get("content") or []
    logprobs = [t["logprob"] for t in lp_content]
    if len(logprobs) != len(resp_ids):
        raise HTTPException(
            502,
            f"logprobs ({len(logprobs)}) misaligned with sampled tokens "
            f"({len(resp_ids)}); refusing to record this transition.",
        )
    return list(prompt_ids), list(resp_ids), logprobs


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_trajectory_id: str | None = Header(default=None),
):
    body = await request.json()

    if body.get("stream"):
        raise HTTPException(400, "streaming not supported through the "
                            "training proxy; set stream=false")

    # Inject capture flags; route to the newest adapter transparently.
    body["return_token_ids"] = True
    body["logprobs"] = True
    if state["adapter"]:
        body["model"] = state["adapter"]

    r = await _client.post("/v1/chat/completions", json=body)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    data = r.json()

    if x_trajectory_id:
        prompt_ids, resp_ids, logprobs = _extract_tokens(data)
        store.record(x_trajectory_id, Transition(
            prompt_token_ids=prompt_ids,
            response_token_ids=resp_ids,
            logprobs=logprobs,
            adapter_version=state["adapter_version"],
        ))

    # Strip capture fields so downstream agent code sees a vanilla response.
    data.pop("prompt_token_ids", None)
    for c in data.get("choices", []):
        c.pop("token_ids", None)
    return data


# --------------------------------------------------------------------------
# Optional in-process training loop
# --------------------------------------------------------------------------

def _training_loop():
    from .trainer import GRPOTrainer, TrainerConfig

    cfg = TrainerConfig.from_env(vllm_base_url=VLLM_URL)
    trainer = GRPOTrainer(cfg)
    log.info("trainer ready; polling for groups of %d", GROUP_SIZE)
    while True:
        group = store.pop_ready_group()
        if group is None:
            time.sleep(2.0)
            continue
        metrics = trainer.train_on_group(group)
        adapter = trainer.push_adapter()
        state["adapter"] = adapter
        state["adapter_version"] = trainer.version
        log.info("step %d done: %s", trainer.version, metrics)


@app.on_event("startup")
async def maybe_start_trainer():
    if TRAIN:
        threading.Thread(target=_training_loop, daemon=True).start()
