# Runbook: single-GPU TITO training on RunPod

Operational knowledge from bringing up Qwen3.5-4B GRPO training on a 24 GB
RTX 3090. Read this before touching a pod; every section below cost real
debugging time to learn.

## TL;DR

```bash
# on the pod
MODEL=Qwen/Qwen3.5-4B scripts/stack.sh stop     # always start from a clean GPU
MODEL=Qwen/Qwen3.5-4B scripts/stack.sh start
scripts/stack.sh train --rounds 30 --tasks-per-round 4 --group-size 8
scripts/stack.sh status
```

## GPU memory budget (24 GB card, 4B model)

| Component | Steady state | During init |
|---|---|---|
| vLLM weights (bf16) | 8.75 GiB | 8.75 GiB |
| vLLM KV cache + overhead | ~3.5 GiB at `--gpu-memory-utilization 0.50` | — |
| vLLM torch.compile + CUDA-graph profiling | — | **spikes several GiB above the util budget** |
| Trainer weights (bf16) | 8.75 GiB | 8.75 GiB (alloc-warmup does one big 7.83 GiB allocation) |
| Trainer LoRA + AdamW + activations | ~1–2 GiB (grad ckpt on, mb=1) | — |

Verdict: 4B + 4B on 24 GB fits **only** if (a) vLLM inits alone on a clean
GPU, and (b) `--enforce-eager` is used to remove the compile/profiling spike.
Even then headroom is ~1–2 GiB — one long episode can OOM the trainer.
**A 48 GB card (A6000, ~$0.50/hr) removes this entire class of failure** and
is the right call for real runs.

## The three failure modes we hit (and their fixes)

### 1. Ghost GPU processes survive `pkill`

`pkill -f 'vllm serve'` kills the API server but not the `EngineCore`
worker subprocesses. The proxy's trainer thread holds CUDA context through
`/dev/nvidiactl`, which `lsof /dev/nvidia0` does not list. Result: 8–15 GiB
of "invisible" GPU memory and the next vLLM start OOMs.

**Fix:** `fuser -k /dev/nvidia*` kills everything holding any nvidia device.
`stack.sh stop` does this and then *verifies* GPU usage is < 500 MiB.
Never trust a kill without verifying `nvidia-smi` afterwards.

### 2. Phantom readiness — stale server answers the health check

After a partial kill, the *old* vLLM answered `GET /health` while the *new*
one was crashing on a port conflict behind it. The launch sequence saw
"healthy after 1 poll" and proceeded to load the trainer against a server
that was about to disappear.

**Fix:** readiness = `/v1/models` lists the expected model **and**
`nvidia-smi` shows more memory in use than the model's weight size
(~8 GB for 4B bf16). Also: refuse to start on a GPU that is not clean.
Both checks live in `stack.sh` and in the proxy's `_wait_for_vllm` gate.

### 3. Failed rollouts scored as reward 0.0

When vLLM died mid-run, litellm surfaced 500s, the driver caught them and
posted `reward=0.0`. 99 trajectories with zero transitions entered the
store. Training on those would teach the model its (never-sampled) actions
were wrong — pure gradient poison.

**Fix:** the driver now `DELETE /trajectories/{id}` on any rollout
exception and retries the slot with a fresh seed. If a slot fails
`--max-retries + 1` times, the run aborts loudly: that's an infra outage,
not a training signal. Watch `aborted` in `/stats` — a nonzero-but-small
count is normal, a climbing count means the stack is unhealthy.

## Pod provisioning gotchas (RunPod)

- **API-created pods don't start sshd.** The web UI injects your account SSH
  key as the `PUBLIC_KEY` env var, which the image's start script needs to
  launch sshd. Pods created via the API get no such injection — both direct
  and proxy SSH will fail (proxy authenticates at the gateway, then dies
  reaching pod:22). Always pass `PUBLIC_KEY` in the create request's env.
- **Check the host driver before installing.** `pip install vllm` pulls the
  newest torch (cu130 as of vLLM 0.24), but CUDA 13 wheels need driver
  r580+. A host on r550 (CUDA 12.4) fails at engine init with
  "The NVIDIA driver on your system is too old". Fix: install the CUDA-12
  variant wheel from vLLM's GitHub releases with `--no-deps`, then
  force-reinstall torch from the matching PyTorch index:

  ```bash
  wget https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
  pip install --force-reinstall --no-deps vllm-0.24.0+cu129-*.whl
  pip install --force-reinstall torch==2.11.0 torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
  ```

  CUDA 12.x minor versions are driver-compatible with each other (cu128/cu129
  run on r550); the 12→13 major jump is not.
- **PEP 668**: recent images mark system Python externally-managed; every
  pip install needs `--break-system-packages` (fine in a disposable container).
- **Don't trust `pip ... | tail -1; echo RC=$?`** — that captures tail's exit
  code, not pip's. Redirect to a log file and check pip's own status.
- **`restart-pod` wipes the container disk when no network volume is
  attached.** /workspace, pip installs, secrets — all gone; only a mounted
  network volume survives. Restart also reassigns the public SSH port, so
  re-query the pod for the new mapping. Treat restart as re-provisioning
  (keep the setup scripted), or attach a network volume for anything you
  can't recreate. The one thing restart is good for: clearing a leaked CUDA
  context that `kill`/`fuser` can't free (dead pid still holding GiBs in
  nvidia-smi with process name `[Not Found]`).
- **A killed vLLM can leak its CUDA context.** After SIGKILL mid-inference,
  nvidia-smi may show the dead pid holding all its memory indefinitely; the
  driver (r550) never reaps it and no in-container command can. Prefer
  graceful shutdown (SIGTERM first, wait, then SIGKILL); if leaked, restart
  the pod.

## Startup order (why it matters)

```
clean GPU  →  vLLM  →  wait until truly ready  →  proxy+trainer  →  driver
```

vLLM's initialization allocates *more* than its steady-state budget
(profiling, compile, graph capture). If the trainer loads its 8.75 GiB
during that window, one of the two processes dies — and which one dies
determines whether you see a vLLM `RuntimeError: Engine core initialization
failed` or a trainer `torch.OutOfMemoryError`, which look like totally
different bugs but are the same race.

The proxy now enforces this in code: the trainer thread blocks on
`_wait_for_vllm()` until `/v1/models` lists `MODEL` before touching the GPU.

## Secrets

`ANTHROPIC_API_KEY` (user simulator) lives in `/workspace/secrets.env`
(chmod 600), written via the RunPod web terminal — never over the logged
SSH channel, never echoed. To pass it to a child process:

```bash
set -a; source /workspace/secrets.env; set +a   # set -a exports; plain source does NOT
```

`stack.sh train` does this for you.

## Qwen3.5 (hybrid Mamba/attention) model notes

- `attn_implementation="sdpa"` — FA2 crashes on the hybrid attention path.
- `micro_batch_size=1` — batch padding perturbs Mamba recurrent state
  (~0.1 nats/pad token), corrupting logprob ratios. Do not raise it.
- LoRA targets include `in_proj`/`out_proj` for the SSM layers; verify
  coverage on a new variant with
  `{k for k,_ in model.named_modules() if "proj" in k}`.
- vLLM logs `no matching PunicaWrapper ... will be ignored` for the visual
  blocks — harmless for text-only training, but check the *language*
  projections did get LoRA-wrapped or hot-swaps silently change nothing.

## Slow ≠ stalled

A GRPO step can take many minutes: ~100 transitions × two forwards each
(behavior-logprob recompute + gradient pass) at micro-batch 1 on multi-k
token sequences, sharing the GPU with live vLLM decode. The FIRST step is
slower still — Triton kernel autotune and torch warmup happen there.
**Before assuming a stall, check `nvidia-smi`:** utilization pinned near
100% with the trainer's memory elevated means it is computing. A real
trainer crash leaves a `_training_loop` traceback in proxy.log; absence of
step metrics alone proves nothing.

Also note `step` (weights version) vs `served_version` in the metrics: they
diverge when an adapter push fails — the trainer keeps training while vLLM
serves the old policy, rollouts stay tagged with the served version, and
the recompute path automatically treats them as off-policy. A persistent
gap means pushes keep failing; check vLLM health.

## Monitoring a run

| Signal | Where | Healthy |
|---|---|---|
| GPU headroom | `nvidia-smi` | > 1 GiB free at steady state |
| trajectories / transitions | `GET :9000/stats` | transitions grows with trajectories (0 transitions + growing trajectories = vLLM down) |
| `aborted` | `/stats` | small and flat |
| `groups_dropped_degenerate` | `/stats` | < ~30% of groups; higher means tasks are too easy/hard for the policy |
| parity check | proxy.log at step 0 | `parity check ok, max_diff < 0.05`; a RuntimeError here means token capture is broken — do not proceed |
| `clip_frac` | `trainer_metrics.jsonl` | ≪ 1.0; pinned at 1.0 means advantage scale is wrong |
| `n_skipped_overlong` | `trainer_metrics.jsonl` | < 10% of `n_transitions`; higher biases training toward short episodes — raise `max_seq_len` |
| eval reward | `rollout_metrics.jsonl` (`"kind": "eval"`) | rising vs. the round-0 baseline; this is the only number that proves training works |

## Log locations (via stack.sh)

```
/workspace/logs/vllm.log                 vLLM server
/workspace/logs/proxy.log                proxy + trainer thread (parity check output here)
/workspace/logs/train.log                rollout driver
/workspace/logs/trainer_metrics.jsonl    one line per GRPO step
/workspace/logs/rollout_metrics.jsonl    one line per round + per eval
```
