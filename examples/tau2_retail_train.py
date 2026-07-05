"""tau2 retail rollout driver for TITO/GRPO training.

Runs GROUP_SIZE parallel tau2 retail episodes per task using the TITO proxy
for agent token capture. The user simulator calls Claude via Anthropic API
so simulator tokens never enter the trajectory store. After each group of G
rollouts the proxy triggers a GRPO step + LoRA hot-swap automatically.

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
import os
import random
import sys
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


# ── per-rollout function ───────────────────────────────────────────────────

def _one_rollout(
    task,
    proxy_url: str,
    model: str,
    anthropic_api_key: str,
    seed: int,
) -> float:
    """Run one tau2 retail episode; return the composite reward."""
    # Create trajectory in proxy — task.id is the GRPO group key
    r = httpx.post(
        f"{proxy_url}/trajectories",
        json={"task_id": str(task.id)},
        timeout=30,
    )
    r.raise_for_status()
    traj_id: str = r.json()["traj_id"]

    # Build a fresh config per rollout so extra_headers carries this traj_id.
    # litellm passes extra_headers as HTTP headers on every completion call.
    config = TextRunConfig(
        domain="retail",
        agent="llm_agent",
        llm_agent=f"openai/{model}",
        llm_args_agent={
            "api_base": f"{proxy_url}/v1",
            "api_key": "EMPTY",
            "extra_headers": {"X-Trajectory-ID": traj_id},
        },
        llm_user="claude-haiku-4-5-20251001",
        llm_args_user={"api_key": anthropic_api_key},
    )

    orchestrator = build_text_orchestrator(config, task, seed=seed)
    result = run_simulation(orchestrator)

    reward = float(result.reward_info.reward) if result.reward_info else 0.0

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
) -> list[float]:
    """Run GROUP_SIZE rollouts for one task in parallel."""
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(group_size):
            futures.append(
                pool.submit(
                    _one_rollout,
                    task,
                    proxy_url,
                    model,
                    anthropic_api_key,
                    base_seed + i,
                )
            )
        rewards = []
        for f in as_completed(futures):
            try:
                rewards.append(f.result())
            except Exception as exc:
                print(f"  rollout error: {exc}")
                rewards.append(0.0)
    return rewards


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
    args = ap.parse_args()

    if not _ANTHROPIC_KEY:
        sys.exit("ANTHROPIC_API_KEY not set — needed for the tau2 user simulator")

    # Verify proxy is up
    try:
        s = httpx.get(f"{args.proxy}/stats", timeout=10).json()
        print(f"proxy stats: {s}")
    except Exception as exc:
        sys.exit(f"Cannot reach proxy at {args.proxy}: {exc}")

    tasks = get_tasks("retail")
    print(f"loaded {len(tasks)} retail tasks")

    rng = random.Random(args.seed)

    for rnd in range(args.rounds):
        round_tasks = rng.sample(tasks, min(args.tasks_per_round, len(tasks)))
        round_rewards: list[float] = []

        for task in round_tasks:
            base_seed = args.seed + rnd * 1000 + hash(task.id) % 1000
            rewards = run_task_group(
                task=task,
                proxy_url=args.proxy,
                model=args.model,
                anthropic_api_key=_ANTHROPIC_KEY,
                group_size=args.group_size,
                workers=args.workers,
                base_seed=base_seed,
            )
            mean_r = sum(rewards) / len(rewards)
            round_rewards.extend(rewards)
            print(f"  round {rnd} task {task.id}: rewards={rewards} mean={mean_r:.3f}")

        overall = sum(round_rewards) / len(round_rewards) if round_rewards else 0.0
        stats = httpx.get(f"{args.proxy}/stats", timeout=10).json()
        print(
            f"round {rnd}: mean_reward={overall:.3f} "
            f"groups_trained={stats.get('groups_ready', '?')} "
            f"adapter={stats.get('adapter', 'none')}"
        )


if __name__ == "__main__":
    main()
