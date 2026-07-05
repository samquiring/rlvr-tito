"""Toy end-to-end rollout driver: a verifiable multi-turn task through the
TITO proxy. Demonstrates the full agent contract in ~60 lines so you can
verify the pipeline (tokens captured, groups formed, adapter hot-swapped)
before pointing tau2 at it.

Task: binary search. The environment holds a secret integer in [1, 64]; each
agent turn must end with "GUESS: <n>"; the env replies higher/lower. Reward 1
if found within max_turns, else 0. Verifiable, multi-turn, and a base
instruct model succeeds sometimes but not always: exactly the informative
regime GRPO needs.

Run G rollouts per secret (task_id = the secret), in parallel across tasks.

  python toy_agent.py --proxy http://localhost:9000 --group-size 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import random
import re

import httpx
from openai import OpenAI

SYSTEM = ("You are playing a number guessing game. The number is between 1 "
          "and 64. Think briefly, then end EVERY reply with a line "
          "'GUESS: <number>'.")


def rollout(proxy: str, model: str, secret: int, max_turns: int = 7) -> float:
    http = httpx.Client(base_url=proxy, timeout=600)
    traj_id = http.post("/trajectories",
                        json={"task_id": f"secret_{secret}"}).json()["traj_id"]
    client = OpenAI(base_url=f"{proxy}/v1", api_key="EMPTY",
                    default_headers={"X-Trajectory-ID": traj_id})

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Start guessing."}]
    reward = 0.0
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=1.0, max_tokens=256)
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        m = re.search(r"GUESS:\s*(\d+)", text)
        guess = int(m.group(1)) if m else -1
        if guess == secret:
            reward = 1.0
            break
        hint = ("Invalid format. End with 'GUESS: <number>'." if guess < 0
                else ("Higher." if guess < secret else "Lower."))
        messages.append({"role": "user", "content": hint})

    http.post(f"/trajectories/{traj_id}/reward", json={"reward": reward})
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://localhost:9000")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Instruct")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--tasks-per-round", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    rng = random.Random(0)
    with cf.ThreadPoolExecutor(max_workers=32) as pool:
        for rnd in range(args.rounds):
            secrets = [rng.randint(1, 64) for _ in range(args.tasks_per_round)]
            futures = [pool.submit(rollout, args.proxy, args.model, s)
                       for s in secrets for _ in range(args.group_size)]
            rewards = [f.result() for f in cf.as_completed(futures)]
            print(f"round {rnd}: mean reward "
                  f"{sum(rewards) / len(rewards):.3f} over {len(rewards)} rollouts")


if __name__ == "__main__":
    main()
