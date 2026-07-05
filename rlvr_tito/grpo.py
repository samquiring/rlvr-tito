"""GRPO core math. Pure torch, no framework dependencies.

Design decisions, made explicit:

- Trajectory-level sparse reward, gamma = 1.0: every response token in every
  transition of a trajectory carries that trajectory's group advantage.
- Token-mean aggregation across the batch (Dr.GRPO / DAPO style) rather than
  per-sequence mean, to avoid length bias in multi-turn settings.
- Asymmetric clipping (clip-higher) supported, DAPO-style.
- Importance ratio is computed against the *behavior* logprobs captured at
  sampling time by vLLM. This is what makes training correct even when the
  serving adapter was updated mid-collection: transitions are mildly
  off-policy and the ratio + clip handles it.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-relative advantage: z-score of terminal rewards within one
    group of G rollouts of the same task.

    Returns zeros for degenerate groups (all rewards equal); callers should
    drop those groups before training to avoid wasting a step.
    """
    r = rewards.to(torch.float32)
    std = r.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(r)
    return (r - r.mean()) / (std + eps)


def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-token log p(label) from logits. logits: (B, T, V), labels: (B, T)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def grpo_loss(
    logp_new: torch.Tensor,      # (B, T) current policy logprobs of taken tokens
    logp_behavior: torch.Tensor,  # (B, T) logprobs captured at sampling time
    advantages: torch.Tensor,     # (B,) one scalar per sample (transition)
    response_mask: torch.Tensor,  # (B, T) 1.0 on response tokens, 0.0 elsewhere
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_coef: float = 0.0,
    logp_ref: torch.Tensor | None = None,  # (B, T) reference policy, optional
) -> tuple[torch.Tensor, dict]:
    """Clipped surrogate over response tokens only.

    Returns (scalar loss, metrics dict). Metrics are detached floats.
    """
    adv = advantages.unsqueeze(-1)                      # (B, 1) broadcast over T
    log_ratio = logp_new - logp_behavior
    ratio = torch.exp(log_ratio)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv
    pg = -torch.minimum(unclipped, clipped)

    if kl_coef > 0.0 and logp_ref is not None:
        # k3 estimator: unbiased, low-variance, always >= 0
        lr = logp_ref - logp_new
        kl = torch.exp(lr) - lr - 1.0
        pg = pg + kl_coef * kl

    denom = response_mask.sum().clamp(min=1.0)
    loss = (pg * response_mask).sum() / denom

    with torch.no_grad():
        clip_frac = (((ratio > 1.0 + clip_high) | (ratio < 1.0 - clip_low)).float()
                     * response_mask).sum() / denom
        metrics = {
            "loss": float(loss),
            "ratio_mean": float((ratio * response_mask).sum() / denom),
            "clip_frac": float(clip_frac),
            "response_tokens": float(denom),
        }
    return loss, metrics
