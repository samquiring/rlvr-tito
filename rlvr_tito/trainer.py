"""GRPO trainer: transitions -> padded batches -> LoRA update -> vLLM hot-swap.

Runs in the same process as the proxy (background thread) or standalone.
Training GPUs must be separate from vLLM's (set CUDA_VISIBLE_DEVICES for each
process accordingly).

Weight sync uses vLLM's runtime LoRA loading:
  - launch vLLM with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
  - after each update we save the adapter and POST /v1/load_lora_adapter
  - the proxy then routes new requests to the new adapter name

This is the ART-style serve-while-training pattern. It is deliberately the
simplest correct sync mechanism; swap in NCCL weight broadcast (verl/slime
style) only when LoRA becomes the bottleneck.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from .grpo import gather_logprobs, group_advantages, grpo_loss
from .store import Trajectory, Transition

log = logging.getLogger("rlvr_tito.trainer")


@dataclass
class TrainerConfig:
    model_name: str = "Qwen/Qwen3.5-9B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    lr: float = 1e-6
    clip_low: float = 0.2
    clip_high: float = 0.28
    kl_coef: float = 0.0               # 0 disables ref model entirely
    max_seq_len: int = 32768
    micro_batch_size: int = 1
    grad_clip: float = 1.0
    adapter_dir: str = "./adapters"
    vllm_base_url: str = "http://localhost:8000"
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True

    @classmethod
    def from_env(cls, **overrides) -> "TrainerConfig":
        """Build config from environment variables (proxy entrypoint)."""
        env = {
            "model_name": os.environ.get("MODEL"),
            "device": os.environ.get("DEVICE"),
            "dtype": os.environ.get("DTYPE"),
            "attn_implementation": os.environ.get("ATTN_IMPL"),
            "adapter_dir": os.environ.get("ADAPTER_DIR"),
            "lr": float(os.environ["LR"]) if "LR" in os.environ else None,
            "lora_rank": int(os.environ["LORA_RANK"]) if "LORA_RANK" in os.environ else None,
            "gradient_checkpointing":
                os.environ.get("GRAD_CKPT", "1") == "1" if "GRAD_CKPT" in os.environ else None,
        }
        kwargs = {k: v for k, v in env.items() if v is not None}
        kwargs.update(overrides)
        return cls(**kwargs)


def collate_transitions(
    transitions: list[Transition],
    advantages: list[float],
    pad_token_id: int,
    max_seq_len: int,
) -> dict[str, torch.Tensor]:
    """Pad transitions into a batch. Each row: [prompt_ids | response_ids].

    Loss mask covers response tokens only. Behavior logprobs are placed
    aligned with response positions. Prompt tokens contribute context, never
    gradient. No re-tokenization happens anywhere in this function: ids in,
    ids out.
    """
    rows, resp_masks, behav_lps, advs = [], [], [], []
    for tr, adv in zip(transitions, advantages):
        ids = tr.prompt_token_ids + tr.response_token_ids
        if len(ids) > max_seq_len:
            # Truncate from the LEFT of the prompt: response tokens are the
            # gradient carriers and must survive intact.
            overflow = len(ids) - max_seq_len
            if overflow >= len(tr.prompt_token_ids):
                log.warning("transition longer than max_seq_len even without "
                            "prompt; skipping")
                continue
            ids = ids[overflow:]
            prompt_len = len(tr.prompt_token_ids) - overflow
        else:
            prompt_len = len(tr.prompt_token_ids)
        mask = [0.0] * prompt_len + [1.0] * len(tr.response_token_ids)
        lp = [0.0] * prompt_len + list(tr.logprobs)
        rows.append(ids)
        resp_masks.append(mask)
        behav_lps.append(lp)
        advs.append(adv)

    if not rows:
        return {}
    T = max(len(r) for r in rows)

    def pad(seqs, value):
        return torch.tensor([s + [value] * (T - len(s)) for s in seqs])

    return {
        "input_ids": pad(rows, pad_token_id).long(),
        "attention_mask": pad([[1] * len(r) for r in rows], 0).long(),
        "response_mask": pad(resp_masks, 0.0).float(),
        "behavior_logprobs": pad(behav_lps, 0.0).float(),
        "advantages": torch.tensor(advs, dtype=torch.float32),
    }


def flatten_group(group: list[Trajectory]) -> tuple[list[Transition], list[float]]:
    """Trajectory-level advantage broadcast to every transition (gamma=1.0)."""
    rewards = torch.tensor([t.reward for t in group])
    advs = group_advantages(rewards)
    transitions, per_transition_adv = [], []
    for traj, a in zip(group, advs):
        for tr in traj.transitions:
            transitions.append(tr)
            per_transition_adv.append(float(a))
    return transitions, per_transition_adv


class GRPOTrainer:
    def __init__(self, cfg: TrainerConfig):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.version = 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=getattr(torch, cfg.dtype),
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device,
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()
        self.model = get_peft_model(base, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_target_modules),
            task_type="CAUSAL_LM",
        ))
        self.opt = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=cfg.lr)

    # -- one GRPO step on one group ------------------------------------------

    def train_on_group(self, group: list[Trajectory]) -> dict:
        transitions, advs = flatten_group(group)
        batch = collate_transitions(
            transitions, advs, self.tokenizer.pad_token_id or 0,
            self.cfg.max_seq_len)
        if not batch:
            return {}

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        B = batch["input_ids"].shape[0]
        mb = self.cfg.micro_batch_size
        total_metrics: dict = {}

        for i in range(0, B, mb):
            sl = slice(i, i + mb)
            ids = batch["input_ids"][sl].to(self.cfg.device)
            attn = batch["attention_mask"][sl].to(self.cfg.device)
            rmask = batch["response_mask"][sl].to(self.cfg.device)
            blp = batch["behavior_logprobs"][sl].to(self.cfg.device)
            adv = batch["advantages"][sl].to(self.cfg.device)

            # Standard causal shift: logits at t predict token t+1.
            logits = self.model(input_ids=ids, attention_mask=attn).logits
            logp = gather_logprobs(logits[:, :-1], ids[:, 1:])
            loss, m = grpo_loss(
                logp_new=logp,
                logp_behavior=blp[:, 1:],
                advantages=adv,
                response_mask=rmask[:, 1:],
                clip_low=self.cfg.clip_low,
                clip_high=self.cfg.clip_high,
            )
            (loss * (min(mb, B - i) / B)).backward()
            total_metrics = m  # last microbatch's metrics; extend as needed

        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.cfg.grad_clip)
        self.opt.step()
        return total_metrics

    # -- weight sync -----------------------------------------------------------

    def push_adapter(self) -> str:
        """Save the LoRA adapter and hot-load it into the running vLLM server.
        Returns the adapter name the proxy should route to."""
        import httpx

        self.version += 1
        name = f"policy_v{self.version}"
        path = os.path.abspath(os.path.join(self.cfg.adapter_dir, name))
        self.model.save_pretrained(path)
        r = httpx.post(
            f"{self.cfg.vllm_base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": path},
            timeout=120,
        )
        r.raise_for_status()
        log.info("loaded %s into vLLM", name)
        return name
