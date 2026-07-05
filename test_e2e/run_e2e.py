"""Single-process E2E: mock vLLM, TITO proxy (with trainer thread), and the
password-game agent, all in one torch runtime. Exists because this test box
has 3GB RAM; the code under test (rlvr_tito.*) is unmodified.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# env BEFORE importing the proxy (it reads env at import time)
os.environ.update({
    "MODEL_DIR": "./tiny-qwen3",
    "VLLM_URL": "http://127.0.0.1:8000",
    "TRAIN": "1", "GROUP_SIZE": "8",
    "MODEL": "./tiny-qwen3", "DEVICE": "cpu", "DTYPE": "float32",
    "ATTN_IMPL": "sdpa", "GRAD_CKPT": "0",
    "LR": os.environ.get("LR", "3e-3"),
    "LORA_RANK": "16", "ADAPTER_DIR": "./adapters",
})

import uvicorn
import httpx

import mock_vllm
from rlvr_tito import proxy as tito_proxy


def serve(app, port):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()


serve(mock_vllm.app, 8000)
serve(tito_proxy.app, 9000)

http = httpx.Client(base_url="http://127.0.0.1:9000", timeout=300)
for _ in range(60):
    try:
        http.get("/stats")
        httpx.get("http://127.0.0.1:8000/health", timeout=5)
        break
    except Exception:
        time.sleep(1)
print("servers up", flush=True)


def rollout(model="base"):
    traj_id = http.post("/trajectories", json={"task_id": "say7"}).json()["traj_id"]
    headers = {"X-Trajectory-ID": traj_id}
    messages = [{"role": "user", "content": "say READY"}]
    for turn in range(2):
        r = http.post("/v1/chat/completions", headers=headers, json={
            "model": model, "messages": messages,
            "temperature": 1.0, "max_tokens": 12})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": text})
        if turn == 0:
            messages.append({"role": "user", "content": "now say the number 7"})
    reward = 1.0 if "7" in messages[-1]["content"] else 0.0
    http.post(f"/trajectories/{traj_id}/reward", json={"reward": reward})
    return reward


ROUNDS = int(os.environ.get("ROUNDS", "12"))
PER_ROUND = int(os.environ.get("PER_ROUND", "32"))
history = []
for rnd in range(ROUNDS):
    rewards = [rollout() for _ in range(PER_ROUND)]
    mean = sum(rewards) / len(rewards)
    history.append(mean)
    s = http.get("/stats").json()
    print(f"round {rnd:02d}  mean_reward={mean:.3f}  adapter={s['adapter']}  "
          f"groups_trained={s['groups_ready']}  "
          f"dropped={s['groups_dropped_degenerate']}", flush=True)
    time.sleep(2)  # let the trainer thread drain the queue

first3, last3 = sum(history[:3]) / 3, sum(history[-3:]) / 3
print(f"first3={first3:.3f}  last3={last3:.3f}")
assert last3 > first3 + 0.15, "reward did not climb"
print("E2E LEARNING TEST: PASS")
