"""TITO trajectory store.

The data model is transition-based, not conversation-based, and that choice is
what makes Qwen3/3.5 chat templates safe to train on:

Each agent turn becomes one Transition holding
  - prompt_token_ids : the EXACT token ids vLLM conditioned on for this turn
                       (as rendered by vLLM's chat template at that moment)
  - response_token_ids: the EXACT token ids the model sampled
  - logprobs          : per-token behavior logprobs captured at sampling time

Because the Qwen3-family template strips <think> blocks from *previous*
assistant turns, the conversation re-rendered at turn N+1 is not the token
sequence sampled at turn N. Whole-conversation single-sample training can
therefore never mask correctly. Per-turn transitions sidestep this entirely:
whatever the template did, prompt_token_ids IS what the model saw, and
response_token_ids IS what it sampled. Nothing is ever re-tokenized.

Trajectories group transitions; a terminal reward arrives once per trajectory;
GRPO groups collect G trajectories of the same task_id.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Transition:
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprobs: list[float]              # behavior logprobs, len == response tokens
    adapter_version: int               # which policy version sampled this
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if len(self.logprobs) != len(self.response_token_ids):
            raise ValueError(
                f"logprobs ({len(self.logprobs)}) and response tokens "
                f"({len(self.response_token_ids)}) misaligned; refusing to "
                "store a transition that would silently corrupt gradients."
            )


@dataclass
class Trajectory:
    traj_id: str
    task_id: str
    transitions: list[Transition] = field(default_factory=list)
    reward: float | None = None

    @property
    def complete(self) -> bool:
        return self.reward is not None


class GroupStore:
    """Thread-safe store keyed by task_id. A group becomes trainable when
    group_size completed trajectories of the same task exist."""

    def __init__(self, group_size: int = 8, drop_degenerate: bool = True):
        self.group_size = group_size
        self.drop_degenerate = drop_degenerate
        self._traj: dict[str, Trajectory] = {}
        self._completed: dict[str, list[Trajectory]] = {}
        self._lock = threading.Lock()
        self.stats = {"trajectories": 0, "transitions": 0, "aborted": 0,
                      "groups_ready": 0, "groups_dropped_degenerate": 0}

    # -- rollout-time API (called by the proxy) ------------------------------

    def start_trajectory(self, task_id: str, traj_id: str | None = None) -> str:
        traj_id = traj_id or uuid.uuid4().hex
        with self._lock:
            self._traj[traj_id] = Trajectory(traj_id=traj_id, task_id=task_id)
            self.stats["trajectories"] += 1
        return traj_id

    def record(self, traj_id: str, transition: Transition) -> None:
        with self._lock:
            traj = self._traj.get(traj_id)
            if traj is None:
                raise KeyError(f"unknown trajectory {traj_id}")
            if traj.complete:
                raise ValueError(f"trajectory {traj_id} already completed")
            traj.transitions.append(transition)
            self.stats["transitions"] += 1

    def complete(self, traj_id: str, reward: float) -> None:
        with self._lock:
            traj = self._traj.pop(traj_id, None)
            if traj is None:
                raise KeyError(f"unknown trajectory {traj_id}")
            traj.reward = float(reward)
            self._completed.setdefault(traj.task_id, []).append(traj)

    def abort(self, traj_id: str) -> None:
        """Discard a trajectory whose rollout failed for infrastructure reasons
        (LLM 500s, network errors, simulator crashes).

        This exists so drivers never have to post reward=0.0 for a rollout that
        failed before the episode could be judged. A zero reward from a dead
        vLLM is indistinguishable from a zero reward from bad policy behavior,
        and training on it teaches the model that whatever it sampled was wrong.
        Aborted trajectories leave the group unfilled; the driver retries the
        rollout to fill the slot instead."""
        with self._lock:
            if self._traj.pop(traj_id, None) is None:
                raise KeyError(f"unknown trajectory {traj_id}")
            self.stats["aborted"] += 1

    # -- trainer-side API -----------------------------------------------------

    def pop_ready_group(self) -> list[Trajectory] | None:
        """Return one full group of completed trajectories, or None.
        Degenerate groups (all rewards identical -> zero advantage everywhere)
        are dropped and counted, DAPO dynamic-sampling style."""
        with self._lock:
            for task_id, done in list(self._completed.items()):
                if len(done) < self.group_size:
                    continue
                group = done[: self.group_size]
                self._completed[task_id] = done[self.group_size:]
                rewards = {t.reward for t in group}
                if self.drop_degenerate and len(rewards) == 1:
                    self.stats["groups_dropped_degenerate"] += 1
                    continue
                self.stats["groups_ready"] += 1
                return group
        return None
