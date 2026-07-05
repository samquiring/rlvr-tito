# rlvr-tito

RLVR for agentic tool use that actually works on small models, in code you can read.

Most RLVR frameworks are built for the multi-GPU, multi-node case and assume a
dedicated research team maintaining them. If you want to run GRPO on a 0.8B–9B
model using tool-calling agents on a single node — or even a single RTX 3090 —
you either end up fighting infrastructure that was never designed for that scale,
or you discover mid-run that the training signal is silently corrupted.

This library exists because two specific problems make naive multi-turn RLVR on
Qwen3/3.5 produce wrong gradients, and neither VeRL nor SLiM-E address both:

**Problem 1 — Retokenization drift (VeRL).** VeRL decodes completions to text
and re-encodes them to get training token ids. For most tokenizers this round-trips
cleanly, but under Qwen3.5's chat template the round-trip is lossy: the ids you
get back are not the ids the model sampled. The gradient then updates the model on
a sequence it never produced. The fix is to never decode and re-encode at all:
vLLM returns the exact sampled ids alongside the completion text when
`return_token_ids: true` is set (vLLM ≥ 0.11). TITO stores those ids directly
and the trainer consumes them. Text exists only for the agent; the training loop
never tokenizes.

**Problem 2 — Qwen think-stripping (VeRL, SLiM-E).** The Qwen3 family chat
template strips `<think>` blocks from previous assistant turns when rendering the
next prompt. This means the conversation rendered at turn N+1 is not a superset
of the tokens conditioned on at turn N. Any training method that builds one
whole-conversation sample and masks response tokens will get the masking wrong:
the prompt boundaries shift every turn. The fix is transition-based samples:
each turn is its own training example whose prompt ids are exactly whatever vLLM
conditioned on for that turn, captured at inference time. The template can
rewrite history however it likes; every sample is internally exact.

**Why not SLiM-E / SLIME?** SLiM-E is a research system with a large surface area
— custom ray actors, a separate rollout cluster, a weight-sync protocol over NCCL.
It's the right tool when you have 64+ GPUs and a team. For a single node, or for
understanding what the training loop is actually doing, it's too much. rlvr-tito
is ~600 lines with no distributed runtime: one vLLM process, one FastAPI proxy,
one training thread. You can read the whole thing in an afternoon.

**The small-model case.** Frontier RLVR work defaults to 7B+ because that's where
the reward signal is meaningful for complex tasks. But for constrained tool-use
tasks — structured output, API calling, single-step verification — a 0.8B model
with a good reward function learns quickly and runs on a single consumer GPU.
rlvr-tito is tested on Qwen/Qwen3.5-0.8B on an RTX 3090; you don't need a cluster
to run a real training loop.

---

## Architecture

```
your agent harness (tau2 gym, custom loop, anything speaking OpenAI chat)
        |  normal /v1/chat/completions  + X-Trajectory-ID header
        v
  TITO proxy (FastAPI) ------------------------> vllm serve (--enable-lora)
        |  injects return_token_ids + logprobs,       ^
        |  records exact (prompt_ids, response_ids,   | /v1/load_lora_adapter
        |  behavior_logprobs) per turn                |
        v                                             |
  GroupStore (transitions grouped by task_id)         |
        |  G completed trajectories of one task       |
        v                                             |
  GRPO trainer (LoRA, clipped surrogate, token-mean) -+
```

The proxy is the entire integration point. Your agent harness needs one header
change; everything else — token capture, group formation, GRPO steps, LoRA
hot-swap — happens transparently.

## The agent contract (all of it)

```python
traj_id = httpx.post(f"{PROXY}/trajectories", json={"task_id": task_id}).json()["traj_id"]

client = OpenAI(base_url=f"{PROXY}/v1", api_key="EMPTY",
                default_headers={"X-Trajectory-ID": traj_id})
# ... run your agent loop with client.chat.completions.create as usual ...

httpx.post(f"{PROXY}/trajectories/{traj_id}/reward", json={"reward": reward})
```

Submit G rollouts per task_id (default 8). When a full group with non-identical
rewards exists, the trainer takes a GRPO step and hot-swaps the new LoRA adapter
into vLLM; the proxy routes subsequent requests to it automatically. Requests made
against a stale adapter are still trained correctly: the stored behavior logprobs
make the importance ratio exact and the clip bounds the update.

## Running it

On a pod, prefer the stack manager — it handles GPU cleanup, the
vLLM-before-trainer startup order, and readiness checks that plain nohup
commands get wrong (see `docs/RUNBOOK.md` for what goes wrong and why):

```bash
MODEL=Qwen/Qwen3.5-4B scripts/stack.sh start
scripts/stack.sh train --rounds 30 --tasks-per-round 4 --group-size 8
scripts/stack.sh status
scripts/stack.sh stop
```

Manually, single GPU (example: RTX 3090, Qwen3.5-0.8B):

```bash
# 1. vLLM serving the policy
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  vllm serve Qwen/Qwen3.5-0.8B --enable-lora --max-lora-rank 32 --port 8000

# 2. proxy + trainer (same GPU, shares VRAM with vLLM)
TRAIN=1 VLLM_URL=http://localhost:8000 MODEL=Qwen/Qwen3.5-0.8B GROUP_SIZE=8 \
  uvicorn rlvr_tito.proxy:app --port 9000

# 3. run a verifiable toy task end-to-end
python examples/toy_agent.py --proxy http://localhost:9000
```

Multi-GPU split on one node (example: 8x H100, 9B policy):

```bash
# vLLM on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  vllm serve Qwen/Qwen3.5-9B-Instruct --enable-lora --max-lora-rank 32 \
  --tensor-parallel-size 4 --port 8000

# proxy + trainer on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 TRAIN=1 VLLM_URL=http://localhost:8000 \
  MODEL=Qwen/Qwen3.5-9B-Instruct GROUP_SIZE=8 \
  uvicorn rlvr_tito.proxy:app --port 9000

# tau2: see examples/tau2_episode.py
```

Install: `pip install torch transformers peft fastapi uvicorn httpx openai`
(vLLM ≥ 0.11 required on the serving side for `return_token_ids`).

## Verifying token fidelity on your model (do this first)

Before any training run, prove the invariant holds on your exact stack: run one
multi-turn rollout, then for a handful of transitions re-render the conversation
with `tokenizer.apply_chat_template` and diff against the stored `prompt_token_ids`.
On Qwen3.5 with thinking enabled they WILL differ — that is the template rewriting
history, which is precisely why training uses the stored ids and not the re-render.
If you ever change serving stack, template, or model family, rerun this check.

## Extending to tau2 and beyond

The proxy knows nothing about tasks; a "dataset" is any driver that (a) picks
task_ids, (b) runs episodes through the proxy, (c) posts verifiable rewards.
For tau2 that means the gym env with the composite reward
(env × action × communicate; decide explicitly whether NL assertions enter the
training reward). Curriculum lives entirely in the driver: sample task_ids from
your bucketing/adaptive sampler and submit groups.

Human-in-the-loop works the same way: a human clicking pass/fail on G sampled
completions of the same prompt is just a slow reward function posting to
`/trajectories/{id}/reward`. The stored behavior logprobs make the resulting
(very off-policy) updates well-defined.

## What is deliberately NOT here

- No PPO value network, no per-turn reward shaping (add only after measuring
  discriminative power; see the misalignment literature).
- No async dispatcher; rollout parallelism is your driver's thread pool.
- No multi-node training; LoRA on one node is the point of this scale.
- No streaming through the proxy (token capture needs the complete response).

## File map

```
rlvr_tito/grpo.py      advantage + clipped surrogate (pure torch, tested)
rlvr_tito/store.py     Transition/Trajectory/GroupStore (tested)
rlvr_tito/trainer.py   collation, GRPO step, LoRA save + vLLM hot-swap
rlvr_tito/proxy.py     OpenAI-compatible capture proxy + training thread
examples/toy_agent.py  end-to-end verifiable toy task (binary search)
examples/tau2_episode.py  tau2 gym integration template
tests/test_core.py     10 tests over the silent-bug surface
```

## Verified

- `tests/test_core.py`: 10 unit tests over advantage math, clip/gradient
  behavior, collation alignment, truncation, and group bookkeeping.
- `test_e2e/run_e2e.py`: full-loop learning test with a 600K-param random-init
  Qwen3-architecture model served through a mock vLLM. Mean reward 0.22 → 0.97
  across 35 GRPO steps with 35 adapter hot-swaps. CPU-only, single process.
- `test_e2e` drift demo: confirms think content sampled at turn N is absent from
  the re-rendered prompt at turn N+1 on a Qwen3-style template — whole-conversation
  masking is impossible, transitions are required.
- Live run on Qwen/Qwen3.5-0.8B on a single RTX 3090 via vLLM 0.24.0: proxy,
  trainer, and toy agent all run end-to-end with no code changes.
