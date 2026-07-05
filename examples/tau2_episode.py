"""tau2-bench integration sketch.

The proxy contract is agent-framework-agnostic, so wiring tau2 in means:
run tau2's agent loop with the agent LLM pointed at the proxy, and report
tau2's composite reward at episode end. Nothing inside tau2 needs to know
about tokens or GRPO.

Requires: tau2-bench installed with the gym extra (uv sync --extra gym).
Check the AgentGymEnv API against your installed version; it has been
refactored across releases, so treat the env calls below as a template.

The user simulator runs INSIDE the tau2 env and should point at a separate
strong model (its own vLLM server), NOT at the training proxy: you do not
want simulator tokens in the trajectory store.
"""

from __future__ import annotations

import httpx
from openai import OpenAI

PROXY = "http://localhost:9000"
POLICY_MODEL = "Qwen/Qwen3.5-9B-Instruct"


def run_tau2_episode(env, task) -> float:
    """One tau2 episode through the TITO proxy.

    env: a tau2 AgentGymEnv (user simulator + DB inside)
    task: a tau2 task object; task.id becomes the GRPO group key
    """
    http = httpx.Client(base_url=PROXY, timeout=600)
    traj_id = http.post("/trajectories",
                        json={"task_id": str(task.id)}).json()["traj_id"]
    client = OpenAI(base_url=f"{PROXY}/v1", api_key="EMPTY",
                    default_headers={"X-Trajectory-ID": traj_id})

    obs, info = env.reset(task=task)
    messages = list(obs["messages"])          # system + policy + user opener
    tools = info["tools"]                     # OpenAI-format tool schemas
    done, reward = False, 0.0

    while not done:
        resp = client.chat.completions.create(
            model=POLICY_MODEL, messages=messages, tools=tools,
            temperature=1.0, max_tokens=2048)
        msg = resp.choices[0].message

        # Hand the assistant action (tool call or user-facing text) to tau2;
        # the env executes tools / queries the user simulator and returns the
        # observation messages to append. Token capture already happened in
        # the proxy when the completion was created.
        obs, reward, done, info = env.step(msg.model_dump())
        messages.append(msg.model_dump())
        messages.extend(obs.get("new_messages", []))

    # tau2's composite reward (env * action * communicate, NL assertions per
    # your training-reward policy) arrives via the env at termination.
    http.post(f"/trajectories/{traj_id}/reward", json={"reward": float(reward)})
    return float(reward)


if __name__ == "__main__":
    raise SystemExit(
        "Template file: import run_tau2_episode into your rollout driver, "
        "construct AgentGymEnv per the tau2 gym docs, and submit G episodes "
        "per task_id in parallel (see examples/toy_agent.py for the driver "
        "pattern)."
    )
