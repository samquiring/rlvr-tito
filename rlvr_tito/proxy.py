"""TITO proxy: an OpenAI-compatible endpoint that any agent harness can call,
which transparently captures exact token ids + logprobs for RL training.

The agent's contract is deliberately tiny:

  1. POST   /trajectories               {"task_id": "..."} -> {"traj_id": ...}
  2. POST   /v1/chat/completions        as usual, plus header X-Trajectory-ID
  3. POST   /trajectories/{id}/reward   {"reward": 1.0} when the episode ends
  4. DELETE /trajectories/{id}          if the rollout failed for infra reasons
                                        (never post reward=0 for a crash)

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

import json
import logging
import os
import threading
import time

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

from .store import GroupStore, Transition

log = logging.getLogger("rlvr_tito.proxy")

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
BASE_MODEL = os.environ.get("MODEL", "")
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
TRAIN = os.environ.get("TRAIN", "0") == "1"
METRICS_PATH = os.environ.get("METRICS_PATH", "./trainer_metrics.jsonl")

app = FastAPI(title="rlvr-tito proxy")
store = GroupStore(group_size=GROUP_SIZE)
state = {"adapter": None, "adapter_version": 0, "last_step_metrics": {}}
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


@app.delete("/trajectories/{traj_id}")
async def abort_trajectory(traj_id: str):
    """Discard a failed rollout so it never enters a training group."""
    try:
        store.abort(traj_id)
    except KeyError:
        raise HTTPException(404, f"unknown trajectory {traj_id}")
    return {"ok": True}


@app.get("/stats")
async def stats():
    return {
        **store.stats,
        "adapter": state["adapter"],
        "adapter_version": state["adapter_version"],
        "last_step_metrics": state["last_step_metrics"],
    }


# --------------------------------------------------------------------------
# OpenAI-compatible passthrough with token capture
# --------------------------------------------------------------------------

async def _fetch_prompt_ids(messages: list, model: str) -> list[int]:
    """Tokenize the prompt via vLLM's /tokenize endpoint.

    vLLM's chat.completions response does not reliably populate
    prompt_token_ids across versions — it may be absent or an empty list.
    A separate /tokenize call with the exact messages + add_generation_prompt
    gives the canonical prompt ids that match what vLLM conditioned on.
    """
    r = await _client.post("/tokenize", json={
        "model": model,
        "messages": messages,
        "add_generation_prompt": True,
    })
    if r.status_code != 200:
        raise HTTPException(502, f"vLLM /tokenize failed ({r.status_code}): {r.text}")
    return r.json()["tokens"]


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
    elif BASE_MODEL:
        body["model"] = BASE_MODEL

    r = await _client.post("/v1/chat/completions", json=body)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    data = r.json()

    if x_trajectory_id:
        choice = data["choices"][0]

        # Response token ids — field name has drifted across vLLM versions.
        resp_ids: list[int] | None = choice.get("token_ids") or data.get("token_ids")
        if not resp_ids:
            raise HTTPException(
                502,
                "vLLM response lacks token_ids (choice.token_ids). "
                "Ensure vLLM >= 0.11 and return_token_ids is supported. "
                f"Keys present: choice={list(choice)}, top={list(data)}",
            )

        # Prompt token ids — NOT reliably returned by all vLLM versions.
        # Fall back to a separate /tokenize call to get the exact ids that
        # vLLM conditioned on. This is the only source of truth.
        prompt_ids: list[int] = list(
            data.get("prompt_token_ids") or choice.get("prompt_token_ids") or []
        )
        if not prompt_ids:
            prompt_ids = await _fetch_prompt_ids(
                body["messages"], body.get("model", BASE_MODEL)
            )

        # Assert against usage counter — mismatch means template drift.
        usage_prompt = (data.get("usage") or {}).get("prompt_tokens")
        if usage_prompt and len(prompt_ids) != usage_prompt:
            raise HTTPException(
                502,
                f"prompt_token_ids length ({len(prompt_ids)}) != "
                f"usage.prompt_tokens ({usage_prompt}). "
                "Chat template mismatch between /tokenize and vLLM inference.",
            )

        lp_content = (choice.get("logprobs") or {}).get("content") or []
        logprobs = [t["logprob"] for t in lp_content]
        if len(logprobs) != len(resp_ids):
            raise HTTPException(
                502,
                f"logprobs ({len(logprobs)}) misaligned with sampled tokens "
                f"({len(resp_ids)}); refusing to record this transition.",
            )

        store.record(x_trajectory_id, Transition(
            prompt_token_ids=prompt_ids,
            response_token_ids=list(resp_ids),
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

def _wait_for_vllm(timeout: float = 900.0) -> None:
    """Block until vLLM is serving BASE_MODEL. Loading the trainer model onto
    the GPU while vLLM is still initializing is the classic OOM race on a
    shared card: vLLM's profiling spikes above its steady-state budget, so the
    trainer must not allocate until vLLM has settled. This gate enforces the
    startup order in code instead of relying on launch scripts to get it right."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{VLLM_URL}/v1/models", timeout=5)
            if r.status_code == 200:
                served = [m["id"] for m in r.json().get("data", [])]
                if not BASE_MODEL or BASE_MODEL in served:
                    log.info("vLLM ready, serving %s", served)
                    return
                log.warning("vLLM up but serving %s, want %s", served, BASE_MODEL)
        except httpx.HTTPError:
            pass
        time.sleep(10)
    raise RuntimeError(
        f"vLLM at {VLLM_URL} not serving {BASE_MODEL!r} after {timeout}s — "
        "refusing to load the trainer model onto a GPU vLLM may still be "
        "initializing on. Start vLLM first and wait for /v1/models."
    )


def _append_metrics(record: dict) -> None:
    try:
        with open(METRICS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.warning("could not write metrics to %s: %s", METRICS_PATH, exc)


def _training_loop():
    from .trainer import GRPOTrainer, TrainerConfig

    _wait_for_vllm()
    cfg = TrainerConfig.from_env(vllm_base_url=VLLM_URL)
    trainer = GRPOTrainer(cfg)
    log.info("trainer ready; polling for groups of %d", GROUP_SIZE)
    while True:
        group = store.pop_ready_group()
        if group is None:
            time.sleep(2.0)
            continue
        rewards = [t.reward for t in group]
        metrics = trainer.train_on_group(group)

        # A failed push must not kill the loop: sampling continues on the old
        # adapter and the importance ratio absorbs the one-step lag. Retry a
        # few times (vLLM may be busy), then log loudly and move on.
        adapter = state["adapter"]
        for attempt in range(3):
            try:
                adapter = trainer.push_adapter()
                break
            except Exception as exc:
                log.error("push_adapter attempt %d failed: %s", attempt + 1, exc)
                time.sleep(5)
        else:
            log.error("adapter push failed 3x — still serving %s; will retry "
                      "after the next group", adapter)
        state["adapter"] = adapter
        state["adapter_version"] = trainer.version
        state["last_step_metrics"] = metrics

        record = {
            "ts": time.time(),
            "step": trainer.version,
            "task_id": group[0].task_id,
            "rewards": rewards,
            "reward_mean": sum(rewards) / len(rewards),
            "adapter": adapter,
            **metrics,
        }
        _append_metrics(record)
        log.info("step %d done: %s", trainer.version, record)


@app.on_event("startup")
async def maybe_start_trainer():
    if TRAIN:
        threading.Thread(target=_training_loop, daemon=True).start()
