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


def _check_parity_match(
    forward_lps: list[float],
    stored_lps: list[float],
    tol: float,
    idx: int,
) -> None:
    """Raise RuntimeError if forward and stored logprobs diverge beyond tol.

    Pure function so it can be unit-tested without a real model.
    Called by GRPOTrainer.parity_check before the first gradient step.
    """
    if len(forward_lps) != len(stored_lps):
        raise RuntimeError(
            f"Parity check transition {idx}: length mismatch — "
            f"forward={len(forward_lps)}, stored={len(stored_lps)}. "
            "prompt_token_ids has the wrong length."
        )
    max_diff = max(abs(a - b) for a, b in zip(forward_lps, stored_lps))
    if max_diff > tol:
        raise RuntimeError(
            f"Parity check transition {idx}: max |forward − stored| = "
            f"{max_diff:.4f} > {tol}. "
            f"forward[:5]={[f'{x:.3f}' for x in forward_lps[:5]]} "
            f"stored[:5]={[f'{x:.3f}' for x in stored_lps[:5]]}. "
            "Prompt reconstruction or logprob capture has drifted — "
            "importance ratios will be garbage. Aborting before first step."
        )

log = logging.getLogger("rlvr_tito.trainer")


@dataclass
class TrainerConfig:
    model_name: str = "Qwen/Qwen3.5-9B-Instruct"
    lora_rank: int = 32
    lora_alpha: int = 64
    # Qwen3.5 is a hybrid attention/linear-attention model. The attention
    # projections (q/k/v/o) and MLP projections (gate/up/down) are listed
    # below. If your variant also has SSM or linear-attention layers, their
    # projections (commonly in_proj, out_proj, x_proj, dt_proj) are NOT
    # covered here and LoRA will leave those weights frozen — verify with
    # `{k for k,_ in model.named_modules() if "proj" in k}` before training.
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "out_proj",               # linear/SSM layers in hybrid archs
    )
    lr: float = 1e-6
    clip_low: float = 0.2
    clip_high: float = 0.28
    kl_coef: float = 0.0               # 0 disables ref model entirely
    max_seq_len: int = 32768
    # DO NOT raise micro_batch_size above 1 for Mamba/hybrid models without
    # re-running parity_check. Batch padding perturbs Mamba recurrent state
    # by ~0.1 nats per pad token — enough to corrupt logprob ratios. At mb=1
    # each forward sees exactly one sequence so collation padding never enters
    # a forward pass and this issue does not arise.
    micro_batch_size: int = 1
    grad_clip: float = 1.0
    adapter_dir: str = "./adapters"
    vllm_base_url: str = "http://localhost:8000"
    device: str = "cuda"
    dtype: str = "bfloat16"
    # Keep sdpa. FA2 crashes on hybrid attention (IMMA kernel path is
    # incompatible) even at sequence length 64. Do not revert to flash_attention_2.
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
            # Skip rather than left-truncate. Left-truncation removes tokens
            # from the start of the prompt, which for multi-turn agents is the
            # system message + tool definitions — the text the reward function
            # was designed against. Silently removing it corrupts the reward
            # signal in ways that are very hard to debug. Raise max_seq_len
            # (Qwen3.5 supports up to 128K) or shorten episode max_steps.
            log.warning(
                "skipping transition: len=%d > max_seq_len=%d. "
                "Raise max_seq_len rather than left-truncating — truncation "
                "removes the system prompt and tool definitions for long episodes.",
                len(ids), max_seq_len,
            )
            continue
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

    # -- on-policy parity gate -----------------------------------------------

    def parity_check(
        self, transitions: list[Transition], tol: float = 0.05, n: int = 3
    ) -> None:
        """Assert trainer forward-pass logprobs ≈ stored behavior logprobs.

        Must be called before the first gradient step (version == 0), when
        LoRA B-matrices are zero so the trainer is equivalent to the base
        model that vLLM was serving. Any mismatch means wrong prompt_token_ids
        or a model/template mismatch, and every importance ratio would be
        garbage. Better to abort loudly here than to diverge silently.

        Only checks adapter_version==0 transitions (base-model rollouts).
        """
        candidates = [t for t in transitions if t.adapter_version == 0][:n]
        if not candidates:
            log.warning("parity_check: no adapter_version==0 transitions; skipping")
            return

        self.model.eval()
        with torch.no_grad():
            for i, tr in enumerate(candidates):
                ids = torch.tensor(
                    [tr.prompt_token_ids + tr.response_token_ids],
                    dtype=torch.long,
                    device=self.cfg.device,
                )
                logits = self.model(input_ids=ids).logits
                logp = gather_logprobs(logits[:, :-1], ids[:, 1:])
                # Response token logprobs in the causal-shifted view start at
                # position prompt_len-1 (predicting ids[prompt_len]).
                p = len(tr.prompt_token_ids)
                forward_lps = logp[0, p - 1 : p - 1 + len(tr.response_token_ids)].tolist()
                _check_parity_match(forward_lps, list(tr.logprobs), tol, i)
                log.info("parity check %d/%d ok  max_diff=%.4f", i + 1, len(candidates),
                         max(abs(a - b) for a, b in zip(forward_lps, tr.logprobs)))
        self.model.train()

    # -- one GRPO step on one group ------------------------------------------

    def train_on_group(self, group: list[Trajectory]) -> dict:
        transitions, advs = flatten_group(group)

        if self.version == 0:
            self.parity_check(transitions)

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

        # Global token count over the full batch (same causal shift as grpo_loss).
        # Computed once here so each micro-batch's backward contributes
        # (local_pg_sum / global_tokens) — a true token-mean over all samples.
        # Scaling a local token-mean by (local_samples / B) instead would mix
        # per-token means with sample fractions, making the effective per-token
        # weight depend on micro-batch size.
        global_tokens = float(batch["response_mask"][:, 1:].sum().clamp(min=1))

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
            # grpo_loss returns a local token-mean; multiply by local token
            # count to recover the sum, then divide by global_tokens so the
            # accumulated gradient equals a token-mean over the full batch.
            local_tokens = float(rmask[:, 1:].sum().clamp(min=1))
            (loss * (local_tokens / global_tokens)).backward()
            total_metrics = m

        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.cfg.grad_clip)
        self.opt.step()
        return total_metrics

    # -- weight sync -----------------------------------------------------------

    def push_adapter(self) -> str:
        """Save the LoRA adapter and hot-load it into the running vLLM server.
        Returns the adapter name the proxy should route to.

        self.version is bumped only after the vLLM POST succeeds. If either
        save_pretrained or the POST fails, self.version is unchanged so the
        proxy's adapter_version tag stays consistent with what vLLM is actually
        serving. A partial save is overwritten on the next successful push.
        """
        import httpx

        candidate = self.version + 1
        name = f"policy_v{candidate}"
        path = os.path.abspath(os.path.join(self.cfg.adapter_dir, name))
        self.model.save_pretrained(path)
        r = httpx.post(
            f"{self.cfg.vllm_base_url}/v1/load_lora_adapter",
            json={"lora_name": name, "lora_path": path},
            timeout=120,
        )
        r.raise_for_status()          # raises on failure; version NOT bumped yet
        self.version = candidate      # commit only after vLLM confirms load
        log.info("loaded %s into vLLM", name)
        return name
