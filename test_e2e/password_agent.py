"""E2E test driver: multi-turn 'password game' through the full stack.

Turn 1: user says 'say READY' -> agent replies (anything accepted)
        env replies: 'now say the number 7'
Turn 2: agent replies -> reward 1.0 iff the token '7' appears.

With a random-init tiny model, '7' appears by chance sometimes (small vocab),
so groups start mixed and GRPO has signal from step one. Success = mean
reward climbing decisively above its starting level.
"""

import argparse
import json
import time

import httpx


def rollout(http: httpx.Client, model: str) -> float:
    traj_id = http.post("/trajectories",
                        json={"task_id": "say7"}).json()["traj_id"]
    headers = {"X-Trajectory-ID": traj_id}
    messages = [{"role": "user", "content": "say READY"}]

    for turn in range(2):
        r = http.post("/v1/chat/completions", headers=headers, json={
            "model": model, "messages": messages,
            "temperature": 1.0, "max_tokens": 12,
        })
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": text})
        if turn == 0:
            messages.append(
                {"role": "user", "content": "now say the number 7"})

    reward = 1.0 if "7" in messages[-1]["content"] else 0.0
    http.post(f"/trajectories/{traj_id}/reward", json={"reward": reward})
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", default="http://localhost:9000")
    ap.add_argument("--model", default="base")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--groups-per-round", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=12)
    args = ap.parse_args()

    http = httpx.Client(base_url=args.proxy, timeout=300)
    history = []
    for rnd in range(args.rounds):
        rewards = [rollout(http, args.model)
                   for _ in range(args.group_size * args.groups_per_round)]
        mean = sum(rewards) / len(rewards)
        stats = http.get("/stats").json()
        history.append(mean)
        print(f"round {rnd:02d}  mean_reward={mean:.3f}  "
              f"adapter={stats['adapter']}  "
              f"groups_trained={stats['groups_ready']}  "
              f"dropped_degenerate={stats['groups_dropped_degenerate']}",
              flush=True)
        time.sleep(1)  # let the trainer thread drain

    print(json.dumps({"history": history}))
    first3 = sum(history[:3]) / 3
    last3 = sum(history[-3:]) / 3
    print(f"first3={first3:.3f} last3={last3:.3f}")
    assert last3 > first3 + 0.15, "reward did not climb; loop not learning"
    print("E2E LEARNING TEST: PASS")


if __name__ == "__main__":
    main()
