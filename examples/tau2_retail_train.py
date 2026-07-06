"""tau2 retail rollout driver for TITO/GRPO training.

Runs GROUP_SIZE parallel tau2 retail episodes per task using the TITO proxy
for agent token capture. The user simulator calls Claude via Anthropic API
so simulator tokens never enter the trajectory store. After each group of G
rollouts the proxy triggers a GRPO step + LoRA hot-swap automatically.

Rollout hygiene: a rollout that crashes (LLM 500, network error, simulator
exception) is ABORTED at the proxy (DELETE /trajectories/{id}) and retried —
never scored 0.0. A zero reward from a dead vLLM is indistinguishable from a
zero reward from bad policy behavior, and training on it teaches the model
that whatever it sampled was wrong. If every rollout in a group fails, the
run stops: that is an infrastructure outage, not a training signal.

Eval: tasks are split deterministically into train/eval before the run
(every Nth task by sorted id is held out). The eval set is run at round 0
(base-model baseline) and every --eval-every rounds, WITHOUT trajectory
recording, so eval episodes never contribute gradient. Reward deltas on the
eval set are the only evidence the model is actually improving.

Usage (pod):

  export ANTHROPIC_API_KEY=...
  export TAU2_PATH=/workspace/tau2-bench   # if tau2 not pip-installed
  cd /workspace/rlvr-tito

  PROXY_URL=http://localhost:9000 MODEL=Qwen/Qwen3.5-4B \\
    python examples/tau2_retail_train.py \\
    --rounds 30 --tasks-per-round 4 --group-size 8 --workers 4

Usage (local, proxy on pod via SSH tunnel):

  ssh -L 9000:localhost:9000 user@pod &
  TAU2_PATH=/Users/you/tau2-bench PROXY_URL=http://localhost:9000 \\
    python examples/tau2_retail_train.py ...

The agent LLM routes through the proxy as  openai/MODEL  with
X-Trajectory-ID injected per rollout via litellm's extra_headers.
The proxy overwrites the model field anyway (BASE_MODEL env var), so
the model string here is informational — keep it consistent with what
vLLM is serving.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# ── tau2-bench path ────────────────────────────────────────────────────────
_tau2_path = os.environ.get("TAU2_PATH", "")
if _tau2_path and _tau2_path not in sys.path:
    sys.path.insert(0, os.path.join(_tau2_path, "src"))

from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.runner.build import build_text_orchestrator  # noqa: E402
from tau2.runner.helpers import get_tasks  # noqa: E402
from tau2.runner.simulation import run_simulation  # noqa: E402

# ── defaults from environment ──────────────────────────────────────────────
_PROXY_DEFAULT = os.environ.get("PROXY_URL", "http://localhost:9000")
_MODEL_DEFAULT = os.environ.get("MODEL", "Qwen/Qwen3.5-4B")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_USER_MODEL = os.environ.get("USER_SIM_MODEL", "claude-haiku-4-5-20251001")


class RolloutError(RuntimeError):
    """A rollout failed for infrastructure reasons (already aborted at proxy)."""


# ── per-rollout function ───────────────────────────────────────────────────

def _one_rollout(
    task,
    proxy_url: str,
    model: str,
    anthropic_api_key: str,
    seed: int,
    record: bool = True,
) -> float:
    """Run one tau2 retail episode; return the composite reward.

    record=True registers a trajectory with the proxy so agent tokens are
    captured for training. record=False runs a pure eval episode: no
    X-Trajectory-ID header, so the proxy forwards without storing anything.
    """
    traj_id: str | None = None
    extra_headers: dict[str, str] = {}
    if record:
        r = httpx.post(
            f"{proxy_url}/trajectories",
            json={"task_id": str(task.id)},
            timeout=30,
        )
        r.raise_for_status()
        traj_id = r.json()["traj_id"]
        extra_headers["X-Trajectory-ID"] = traj_id

    # Build a fresh config per rollout so extra_headers carries this traj_id.
    # litellm passes extra_headers as HTTP headers on every completion call.
    config = TextRunConfig(
        domain="retail",
        agent="llm_agent",
        llm_agent=f"openai/{model}",
        llm_args_agent={
            "api_base": f"{proxy_url}/v1",
            "api_key": "EMPTY",
            "extra_headers": extra_headers,
        },
        llm_user=_USER_MODEL,
        llm_args_user={"api_key": anthropic_api_key},
    )

    try:
        orchestrator = build_text_orchestrator(config, task, seed=seed)
        result = run_simulation(orchestrator)
    except Exception as exc:
        # Infra failure mid-episode: discard the partial trajectory so it can
        # never enter a training group with a bogus reward.
        if traj_id:
            try:
                httpx.delete(f"{proxy_url}/trajectories/{traj_id}", timeout=30)
            except httpx.HTTPError:
                pass  # proxy down; trajectory stays incomplete and is never trained on
        raise RolloutError(f"task {task.id} seed {seed}: {exc}") from exc

    reward = float(result.reward_info.reward) if result.reward_info else 0.0

    if traj_id:
        # Post reward — triggers GRPO step once G trajectories for this task_id land
        httpx.post(
            f"{proxy_url}/trajectories/{traj_id}/reward",
            json={"reward": reward},
            timeout=30,
        ).raise_for_status()

    return reward


# ── one group (G parallel rollouts of the same task) ──────────────────────

def run_task_group(
    task,
    proxy_url: str,
    model: str,
    anthropic_api_key: str,
    group_size: int,
    workers: int,
    base_seed: int,
    max_retries: int = 2,
) -> tuple[list[float], int]:
    """Run GROUP_SIZE successful rollouts for one task in parallel.

    Failed rollouts are retried with a fresh seed up to max_retries times per
    slot. Returns (rewards, n_failures). Raises if a slot cannot be filled —
    an unfillable group means vLLM/proxy/Anthropic is down, and continuing
    would leave a partial group permanently blocking the task_id queue.
    """
    rewards: list[float] = []
    n_failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(_one_rollout, task, proxy_url, model,
                        anthropic_api_key, base_seed + i): 0
            for i in range(group_size)
        }
        retry_seed = base_seed + group_size  # fresh seeds for retries
        while pending:
            done_iter = as_completed(list(pending))
            fut = next(done_iter)
            attempt = pending.pop(fut)
            try:
                rewards.append(fut.result())
            except Exception as exc:
                n_failures += 1
                print(f"  rollout failed (attempt {attempt + 1}): {exc}")
                if attempt >= max_retries:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"task {task.id}: rollout slot failed {max_retries + 1} "
                        "times — infrastructure is down, not the policy. "
                        "Check vLLM/proxy health before resuming."
                    ) from exc
                pending[pool.submit(
                    _one_rollout, task, proxy_url, model,
                    anthropic_api_key, retry_seed)] = attempt + 1
                retry_seed += 1
    return rewards, n_failures


# ── eval (no recording, no gradient) ───────────────────────────────────────

def run_eval(
    tasks,
    proxy_url: str,
    model: str,
    anthropic_api_key: str,
    workers: int,
    seed: int,
    rollouts_per_task: int = 1,
) -> dict:
    """Run held-out tasks without trajectory recording; return reward summary."""
    results: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for task in tasks:
            for i in range(rollouts_per_task):
                f = pool.submit(_one_rollout, task, proxy_url, model,
                                anthropic_api_key, seed + i, False)
                futs[f] = str(task.id)
        for f in as_completed(futs):
            tid = futs[f]
            try:
                # f.result() must be evaluated BEFORE touching the dict:
                # setdefault-then-append would create an empty entry when
                # result() raises, and empty lists break the mean below.
                r = f.result()
            except Exception as exc:
                print(f"  eval rollout failed for task {tid}: {exc}")
                continue
            results.setdefault(tid, []).append(r)

    all_rewards = [r for rs in results.values() for r in rs]
    return {
        "n_tasks": len(tasks),
        "n_completed": len(all_rewards),
        "reward_mean": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "per_task": {t: sum(rs) / len(rs) for t, rs in sorted(results.items())},
    }


def split_tasks(tasks, eval_every_nth: int) -> tuple[list, list]:
    """Deterministic train/eval split: every Nth task by sorted id is eval.

    Deterministic so the eval set is identical across runs and across
    baseline/final comparisons — a shifting eval set makes deltas meaningless.
    """
    ordered = sorted(tasks, key=lambda t: str(t.id))
    eval_tasks = ordered[::eval_every_nth]
    eval_ids = {str(t.id) for t in eval_tasks}
    train_tasks = [t for t in ordered if str(t.id) not in eval_ids]
    return train_tasks, eval_tasks


# ── main training loop ─────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="tau2 retail TITO training loop")
    ap.add_argument("--proxy", default=_PROXY_DEFAULT, help="TITO proxy URL")
    ap.add_argument("--model", default=_MODEL_DEFAULT, help="vLLM-served model name")
    ap.add_argument("--rounds", type=int, default=20, help="training rounds")
    ap.add_argument("--tasks-per-round", type=int, default=4,
                    help="distinct tasks sampled each round")
    ap.add_argument("--group-size", type=int, default=8,
                    help="rollouts per task (must match proxy GROUP_SIZE)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel threads per task group")
    ap.add_argument("--seed", type=int, default=0, help="base random seed")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="retries per failed rollout before declaring infra down")
    ap.add_argument("--holdout-every", type=int, default=5,
                    help="every Nth task (by sorted id) held out for eval; 0 disables")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="run the eval set every N rounds (plus round 0 baseline)")
    ap.add_argument("--eval-rollouts", type=int, default=1,
                    help="rollouts per eval task")
    ap.add_argument("--metrics-path", default="./rollout_metrics.jsonl",
                    help="JSONL file for per-round and eval metrics")
    args = ap.parse_args()

    if not _ANTHROPIC_KEY:
        sys.exit("ANTHROPIC_API_KEY not set — needed for the tau2 user simulator")

    # Verify proxy is up
    try:
        s = httpx.get(f"{args.proxy}/stats", timeout=10).json()
        print(f"proxy stats: {s}")
    except Exception as exc:
        sys.exit(f"Cannot reach proxy at {args.proxy}: {exc}")

    def log_metrics(record: dict) -> None:
        record["ts"] = time.time()
        with open(args.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    tasks = get_tasks("retail")
    if args.holdout_every > 0:
        train_tasks, eval_tasks = split_tasks(tasks, args.holdout_every)
    else:
        train_tasks, eval_tasks = list(tasks), []
    print(f"loaded {len(tasks)} retail tasks: "
          f"{len(train_tasks)} train, {len(eval_tasks)} eval")

    def maybe_eval(rnd: int) -> None:
        if not eval_tasks or args.eval_every <= 0:
            return
        if rnd % args.eval_every != 0:
            return
        print(f"── eval @ round {rnd} ({len(eval_tasks)} held-out tasks) ──")
        ev = run_eval(eval_tasks, args.proxy, args.model, _ANTHROPIC_KEY,
                      args.workers, args.seed + 90_000 + rnd,
                      args.eval_rollouts)
        adapter = httpx.get(f"{args.proxy}/stats", timeout=10).json().get("adapter")
        print(f"eval @ round {rnd}: reward_mean={ev['reward_mean']:.3f} "
              f"({ev['n_completed']} episodes, adapter={adapter})")
        log_metrics({"kind": "eval", "round": rnd, "adapter": adapter, **ev})

    rng = random.Random(args.seed)

    maybe_eval(0)  # base-model baseline — the number every later eval is judged against

    for rnd in range(args.rounds):
        round_tasks = rng.sample(train_tasks,
                                 min(args.tasks_per_round, len(train_tasks)))
        round_rewards: list[float] = []
        round_failures = 0

        for task in round_tasks:
            base_seed = args.seed + rnd * 1000 + hash(task.id) % 1000
            rewards, n_fail = run_task_group(
                task=task,
                proxy_url=args.proxy,
                model=args.model,
                anthropic_api_key=_ANTHROPIC_KEY,
                group_size=args.group_size,
                workers=args.workers,
                base_seed=base_seed,
                max_retries=args.max_retries,
            )
            mean_r = sum(rewards) / len(rewards)
            round_rewards.extend(rewards)
            round_failures += n_fail
            print(f"  round {rnd} task {task.id}: rewards={rewards} "
                  f"mean={mean_r:.3f} failures={n_fail}")

        overall = sum(round_rewards) / len(round_rewards) if round_rewards else 0.0
        stats = httpx.get(f"{args.proxy}/stats", timeout=10).json()
        print(
            f"round {rnd}: mean_reward={overall:.3f} "
            f"failures={round_failures} "
            f"groups_trained={stats.get('groups_ready', '?')} "
            f"aborted={stats.get('aborted', '?')} "
            f"adapter={stats.get('adapter', 'none')}"
        )
        log_metrics({
            "kind": "train_round", "round": rnd,
            "reward_mean": overall, "rewards": round_rewards,
            "n_failures": round_failures,
            "groups_trained": stats.get("groups_ready"),
            "adapter": stats.get("adapter"),
            "last_step_metrics": stats.get("last_step_metrics", {}),
        })

        maybe_eval(rnd + 1)


if __name__ == "__main__":
    main()
