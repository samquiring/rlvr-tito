# rlvr-tito

A minimal, flexible RLVR training loop for agentic tasks that trains on the raw
tokens the policy actually sampled. Token-In, Token-Out end to end: nothing is
ever re-tokenized, so the retokenization drift and chat-template masking
problems that break naive multi-turn setups (including the Qwen3-family
think-stripping issue) cannot occur by construction.

The architecture is the ART / Agent-Lightning pattern in ~600 lines you fully
control:

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

## Why this solves your VeRL problem

Two independent bugs plague multi-turn RL on Qwen3/3.5, and this design removes
both structurally rather than patching them:

1. Retokenization drift. Decoding a completion to text and re-encoding it does
   not always reproduce the sampled token ids; the gradient then lands on a
   sequence the model never produced. Fix: vLLM's OpenAI endpoints return the
   exact prompt and response token ids when `return_token_ids: true` is set
   (vLLM >= 0.11). The proxy stores those ids; the trainer consumes those ids.
   Text exists only for the agent's benefit.

2. Chat-template history rewriting. The Qwen3-family template strips `<think>`
   blocks from previous assistant turns, so the conversation rendered at turn
   N+1 is not a superset of the tokens sampled at turn N. Whole-conversation
   training samples can never mask correctly. Fix: transition-based samples.
   Each turn is its own training sample whose prompt ids are whatever vLLM
   actually conditioned on for that turn. The template can rewrite history
   however it wants; every sample remains internally exact.

Cost of the transition representation: shared conversation prefixes are
recomputed in the forward pass instead of packed. At 9B with LoRA this is a
throughput tax, not a correctness issue; prefix-packing (SkyRL-Agent style) is
the upgrade path if it starts to hurt.

## The agent contract (all of it)

```python
traj_id = httpx.post(f"{PROXY}/trajectories", json={"task_id": task_id}).json()["traj_id"]

client = OpenAI(base_url=f"{PROXY}/v1", api_key="EMPTY",
                default_headers={"X-Trajectory-ID": traj_id})
# ... run your agent loop with client.chat.completions.create as usual ...

httpx.post(f"{PROXY}/trajectories/{traj_id}/reward", json={"reward": reward})
```

Submit G rollouts per task_id (default 8). When a full group with non-identical
rewards exists, the trainer takes a GRPO step and hot-swaps the new LoRA
adapter into vLLM; the proxy routes subsequent requests to it automatically.
Requests made against a stale adapter are still trained correctly: the stored
behavior logprobs make the importance ratio exact and the clip bounds the
update.

## Running it

GPU split on one node (example: 8x H100, 9B policy):

```bash
# 1. vLLM serving the policy (GPUs 0-3)
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  vllm serve Qwen/Qwen3.5-9B-Instruct --enable-lora --max-lora-rank 32 \
  --tensor-parallel-size 4 --port 8000

# 2. proxy + trainer (GPUs 4-7)
CUDA_VISIBLE_DEVICES=4,5,6,7 TRAIN=1 VLLM_URL=http://localhost:8000 \
  MODEL=Qwen/Qwen3.5-9B-Instruct GROUP_SIZE=8 \
  uvicorn rlvr_tito.proxy:app --port 9000

# 3. sanity: verifiable toy task through the whole loop
python examples/toy_agent.py --proxy http://localhost:9000

# 4. tau2: see examples/tau2_episode.py; the user simulator gets its OWN
#    vLLM server and never touches the proxy.
```

Install: `pip install torch transformers peft fastapi uvicorn httpx openai`
(pin vLLM >= 0.11 on the serving side for return_token_ids).

## Verifying token fidelity on your model (do this first)

Before any training run, prove the invariant holds on your exact stack: run
one multi-turn rollout, then for a handful of transitions re-render the
conversation with `tokenizer.apply_chat_template` and diff against the stored
`prompt_token_ids`. On Qwen3.5 with thinking enabled they WILL differ (that is
the template rewriting history), which is precisely why training uses the
stored ids and not the re-render. If you ever change serving stack, template,
or model family, rerun this check.

## Extending to tau2 and beyond

The proxy knows nothing about tasks; a "dataset" is any driver that (a) picks
task_ids, (b) runs episodes through the proxy, (c) posts verifiable rewards.
For tau2 that means the gym env with the composite reward
(env * action * communicate; decide explicitly whether NL assertions enter the
training reward). Curriculum lives entirely in the driver: sample task_ids
from your bucketing/adaptive sampler and submit groups.

Human-in-the-loop works the same way: a human clicking pass/fail on G sampled
completions of the same prompt is just a slow reward function posting to
/trajectories/{id}/reward. The stored behavior logprobs are what make the
resulting (very off-policy) updates well-defined.

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
- `test_e2e/run_e2e.py`: full-loop learning test. A 600K-param random-init
  Qwen3-architecture model, served through a mock vLLM (same JSON contract:
  return_token_ids, logprobs, /v1/load_lora_adapter), trained live through
  the proxy on a 2-turn verifiable task: mean reward 0.22 -> 0.97 across 35
  GRPO steps with 35 adapter hot-swaps. CPU-only, single process.
- `test_e2e` drift demo: confirms think content sampled at turn N is absent
  from the re-rendered prompt at turn N+1 on a Qwen3-style template, i.e.
  whole-conversation masking is impossible and transitions are required.

Not yet verified (needs your GPUs): real vLLM serving (exact field placement
of token_ids in your pinned version), Qwen3.5 tool-call parsing through the
proxy, and throughput at 9B. Run test_e2e/password_agent.py against real
vLLM as the first smoke test on your cluster.
