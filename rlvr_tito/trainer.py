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
    # Keep ON. Disabling trades ~2x step speed for O(layers x T) activation
    # memory — long-episode backwards then OOM even on a dedicated 48 GB card.
    gradient_checkpointing: bool = True
    # Resume from a previously saved adapters/policy_v{N} checkpoint. The LoRA
    # weights are restored and version continues from N; AdamW moments are NOT
    # restored (fresh optimizer state — acceptable for LoRA-scale RL).
    resume_version: int = 0

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
            "resume_version":
                int(os.environ["RESUME_VERSION"]) if "RESUME_VERSION" in os.environ else None,
            # Defaults ON even when the var is unset; only an explicit
            # GRAD_CKPT=0 disables (and __init__ warns loudly if it does).
            "gradient_checkpointing": os.environ.get("GRAD_CKPT", "1") == "1",
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
    rows, resp_masks, behav_lps, advs, versions = [], [], [], [], []
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
        versions.append(tr.adapter_version)

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
        "adapter_versions": torch.tensor(versions, dtype=torch.long),
    }


def aggregate_step_metrics(rows: list[tuple[dict, float]]) -> dict:
    """Token-weighted mean of per-row grpo_loss metrics.

    Each row's grpo_loss returns means over that row's response tokens.
    Assigning the last row's dict to the step metrics (the old behavior)
    reported loss/ratio_mean/clip_frac of ONE transition as if it were the
    group — a short final row could show clip_frac 0.0 while the group
    clipped heavily. Weighting each row's means by its token count recovers
    the true token-level means over the whole group, consistent with how the
    gradient itself is normalized.
    """
    total = sum(n for _, n in rows)
    if not rows or total <= 0:
        return {}
    agg = {
        k: sum(m[k] * n for m, n in rows) / total
        for k in rows[0][0]
        if k != "response_tokens"
    }
    agg["response_tokens"] = float(total)
    return agg


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
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        # Two version counters, deliberately separate:
        #   version        — how many optimizer steps these weights embody
        #   served_version — which policy_vN vLLM is actually serving
        # They diverge when an adapter push fails: the trainer keeps training
        # while vLLM serves the old policy, and rollouts keep getting tagged
        # with the OLD (served) version. Conflating them made the recompute
        # path rescore old-policy samples under new weights.
        self.version = int(cfg.resume_version)
        self.served_version = 0
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=getattr(torch, cfg.dtype),
            attn_implementation=cfg.attn_implementation,
            device_map=cfg.device,
        )
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            # With LoRA all base params (incl. embeddings) are frozen, so
            # checkpointed segments see no grad-requiring inputs and the
            # backward silently produces no adapter gradients without this.
            base.enable_input_require_grads()
        else:
            log.warning(
                "gradient checkpointing is DISABLED — activation memory grows "
                "with layers x sequence length and long-episode backwards WILL "
                "OOM. Set GRAD_CKPT=1 unless you know exactly why not.")

        if self.version > 0:
            path = os.path.abspath(
                os.path.join(cfg.adapter_dir, f"policy_v{self.version}"))
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"resume_version={self.version} but no checkpoint at {path}")
            self.model = PeftModel.from_pretrained(base, path, is_trainable=True)
            log.info("resumed adapter from %s (version %d)", path, self.version)
        else:
            self.model = get_peft_model(base, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                target_modules=list(cfg.lora_target_modules),
                task_type="CAUSAL_LM",
            ))
        self.opt = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=cfg.lr)

    def _load_adapter_into_vllm(self, name: str, path: str,
                                timeout: float = 120.0) -> None:
        """POST load_lora_adapter, treating "already loaded" as recoverable.

        A lost response to a successful load (or a retry after a partial
        failure) makes vLLM answer "adapter ... has already been loaded" on
        the second attempt. That is not a failure: retry once with
        load_inplace=True so vLLM refreshes the weights from disk, and if
        even that reports already-loaded, the adapter is registered — done.
        """
        import httpx

        def _post(body: dict) -> httpx.Response:
            return httpx.post(
                f"{self.cfg.vllm_base_url}/v1/load_lora_adapter",
                json=body, timeout=timeout)

        r = _post({"lora_name": name, "lora_path": path})
        if r.status_code == 200:
            return
        if "already" in r.text.lower() and "loaded" in r.text.lower():
            log.info("%s reported already loaded; reloading with "
                     "load_inplace=True", name)
            r2 = _post({"lora_name": name, "lora_path": path,
                        "load_inplace": True})
            if r2.status_code == 200 or (
                    "already" in r2.text.lower() and "loaded" in r2.text.lower()):
                return
            r2.raise_for_status()
        r.raise_for_status()

    def register_current_adapter(self, attempts: int = 4) -> str:
        """Load the resumed adapter checkpoint into vLLM WITHOUT bumping the
        version — used at startup after resume_version so serving matches the
        trainer's restored weights before the next push.

        Retries with backoff and a generous timeout: at startup vLLM may be
        busy servicing rollouts from an already-running driver, and the load
        request can sit behind long decodes (httpx.ReadTimeout). If it still
        fails after all attempts, restart the proxy while vLLM is idle.
        """
        import httpx

        name = f"policy_v{self.version}"
        path = os.path.abspath(os.path.join(self.cfg.adapter_dir, name))
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                self._load_adapter_into_vllm(name, path, timeout=300.0)
                self.served_version = self.version
                log.info("registered resumed adapter %s with vLLM", name)
                return name
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                log.warning("resume adapter load attempt %d/%d failed (%s); "
                            "vLLM may be busy with rollouts — retrying in 20s",
                            i + 1, attempts, type(exc).__name__)
                time.sleep(20)
        raise RuntimeError(
            f"could not register resumed adapter {name} after {attempts} "
            "attempts — vLLM is likely saturated with rollout traffic. "
            "Stop the driver and restart the proxy while vLLM is idle."
        ) from last_exc

    # -- one GRPO step on one group ------------------------------------------

    def _forward_response_logprobs(
        self, ids_row: torch.Tensor, resp_len: int
    ) -> torch.Tensor:
        """Forward one unpadded row; return (1, resp_len) logprobs of its last
        resp_len tokens (the response — always the suffix of a transition).

        Uses logits_to_keep so the LM head runs only over the response suffix.
        HF upcasts logits to fp32 inside forward, so materializing them for a
        full 20k-token prompt is ~12 GiB and OOMs the GPU shared with vLLM —
        prompt positions carry no gradient and their logits are pure waste.
        """
        L = ids_row.shape[1]
        assert 0 < resp_len < L, f"resp_len={resp_len} must be in (0, {L})"
        out = self.model(input_ids=ids_row, logits_to_keep=resp_len + 1)
        # logits cover positions L-resp_len-1 .. L-1; logits[t] predicts
        # token t+1, so dropping the last logit aligns them with the last
        # resp_len tokens as labels.
        return gather_logprobs(out.logits[:, :-1], ids_row[:, -resp_len:])

    def _recompute_behavior_logprobs(self, batch: dict) -> dict:
        """Replace stored (vLLM) behavior logprobs with the trainer's own
        forward-pass logprobs for current-version rows, in place.

        For transitions whose adapter_version == self.version, the trainer's
        weights ARE the sampling policy right now (the step for this group
        hasn't happened yet), so recomputing makes the importance ratio start
        at exactly 1 by construction. This removes the numeric bias between
        vLLM's fused hybrid-Mamba kernels and the trainer's torch path —
        measured at median 1e-4 but with a long tail (p99 ≈ 0.07, max ≈ 0.32
        nats on real episodes), which would otherwise leak into gradients as
        up-to-35% ratio errors on tail tokens.

        The stored vLLM logprobs are demoted to a capture-sanity check: the
        MEDIAN |recomputed − stored| over response tokens must stay tiny.
        Kernel noise never moves the median; a wrong prompt reconstruction
        moves every token — so a median above tol means the capture path is
        broken and training must stop. Stale-version rows (sampled under an
        older adapter we no longer have loaded) keep their stored logprobs:
        the tail bias there is bounded by PPO clipping.

        Returns parity stats for step metrics.
        """
        if self.cfg.micro_batch_size != 1:
            raise NotImplementedError(
                "response-suffix forwards assume one unpadded row per forward; "
                "micro_batch_size must stay 1 — batch padding corrupts Mamba "
                "recurrent state (~0.1 nats per pad token), silently poisoning "
                "logprob ratios. Do not raise it for hybrid/Mamba models.")
        # Rows are eligible for recompute only when the trainer's weights ARE
        # the policy that sampled them: the row's tag must match the served
        # version AND served must equal the weights version. After a failed
        # push those diverge (weights stepped, vLLM still serving the old
        # adapter, rollouts still tagged old) — recomputing would rescore
        # old-policy samples under new weights, so such rows keep their
        # stored vLLM logprobs and rely on PPO clipping instead.
        on_policy = self.served_version == self.version
        B = batch["input_ids"].shape[0]
        diffs: list[torch.Tensor] = []
        n_rows = 0
        self.model.eval()
        with torch.no_grad():
            for i in range(B):
                if (not on_policy
                        or int(batch["adapter_versions"][i]) != self.served_version):
                    continue
                true_len = int(batch["attention_mask"][i].sum())
                resp_len = int(batch["response_mask"][i].sum())
                p = true_len - resp_len  # prompt length
                ids_row = batch["input_ids"][i : i + 1, :true_len].to(self.cfg.device)
                lp = self._forward_response_logprobs(ids_row, resp_len).float().cpu()
                # Full-layout behavior slot for response token j is p + j.
                stored = batch["behavior_logprobs"][i, p : p + resp_len]
                diffs.append((lp[0] - stored).abs())
                batch["behavior_logprobs"][i, p : p + resp_len] = lp[0]
                n_rows += 1
        self.model.train()

        if not diffs:
            log.warning("no on-policy rows to recompute (weights v%d, served "
                        "v%d) — training entirely on stored vLLM logprobs for "
                        "this group", self.version, self.served_version)
            return {"n_lp_recomputed_rows": 0}
        d = torch.cat(diffs)
        stats = {
            "n_lp_recomputed_rows": n_rows,
            "parity_median_diff": d.median().item(),
            "parity_p99_diff": d.quantile(0.99).item() if d.numel() > 1 else d.max().item(),
            "parity_max_diff": d.max().item(),
        }
        if stats["parity_median_diff"] > 0.05:
            raise RuntimeError(
                f"Capture-path failure: median |trainer − vLLM| logprob diff = "
                f"{stats['parity_median_diff']:.4f} > 0.05 across {d.numel()} "
                "response tokens. Kernel noise never moves the median — the "
                "prompt reconstruction or logprob capture is broken. Aborting."
            )
        log.info("behavior logprobs recomputed for %d rows: %s", n_rows, stats)
        return stats

    def train_on_group(self, group: list[Trajectory]) -> dict:
        """One GRPO step with OOM recovery.

        A mid-forward OOM leaves partial grads and fragmented reserved blocks
        in the caching allocator; retrying on top of that fails at a fraction
        of the true requirement. Clear both, then retry once — if it OOMs
        again the group genuinely doesn't fit, so skip it loudly rather than
        killing the training loop.
        """
        for attempt in (1, 2):
            try:
                return self._train_on_group_inner(group)
            except torch.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                log.warning(
                    "OOM during GRPO step (attempt %d/2) — cleared grads and "
                    "cache%s", attempt,
                    "; retrying" if attempt == 1 else "")
        log.error("GRPO step OOM'd twice — skipping this group. If this "
                  "recurs, lower max_seq_len or give the trainer its own GPU.")
        return {"oom_skipped_group": 1}

    def _train_on_group_inner(self, group: list[Trajectory]) -> dict:
        transitions, advs = flatten_group(group)

        batch = collate_transitions(
            transitions, advs, self.tokenizer.pad_token_id or 0,
            self.cfg.max_seq_len)
        if not batch:
            log.error(
                "ALL %d transitions in this group exceeded max_seq_len=%d — "
                "the group produced zero gradient. Raise max_seq_len.",
                len(transitions), self.cfg.max_seq_len,
            )
            return {"n_transitions": len(transitions),
                    "n_skipped_overlong": len(transitions)}

        parity_stats = self._recompute_behavior_logprobs(batch)

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        B = batch["input_ids"].shape[0]
        per_row_metrics: list[tuple[dict, float]] = []

        # Global token count over the full batch. Each row's backward
        # contributes (local_pg_sum / global_tokens) so the accumulated
        # gradient is a true token-mean over all transitions in the group —
        # scaling a local token-mean by sample fractions instead would make
        # each token's weight depend on which row it sits in.
        global_tokens = float(batch["response_mask"].sum().clamp(min=1))

        for i in range(B):
            true_len = int(batch["attention_mask"][i].sum())
            resp_len = int(batch["response_mask"][i].sum())
            p = true_len - resp_len
            ids_row = batch["input_ids"][i : i + 1, :true_len].to(self.cfg.device)
            # Response-suffix forward (see _forward_response_logprobs): the
            # prompt's logits are never materialized. Loss tensors are built
            # response-only; grpo_loss math is unchanged.
            logp = self._forward_response_logprobs(ids_row, resp_len)
            blp = batch["behavior_logprobs"][i : i + 1, p : p + resp_len].to(self.cfg.device)
            adv = batch["advantages"][i : i + 1].to(self.cfg.device)
            loss, m = grpo_loss(
                logp_new=logp,
                logp_behavior=blp,
                advantages=adv,
                response_mask=torch.ones_like(logp),
                clip_low=self.cfg.clip_low,
                clip_high=self.cfg.clip_high,
            )
            # grpo_loss returns a local token-mean; recover the sum, then
            # normalize by the global count for a full-batch token-mean.
            (loss * (float(resp_len) / global_tokens)).backward()
            per_row_metrics.append((m, float(resp_len)))

        torch.nn.utils.clip_grad_norm_(
            (p for p in self.model.parameters() if p.requires_grad),
            self.cfg.grad_clip)
        self.opt.step()
        # The trainer's weights have now moved past what vLLM serves; bump
        # the weights version immediately so nothing (recompute eligibility,
        # metrics) mistakes post-step weights for the sampling policy. The
        # SERVED version advances separately, only when push_adapter succeeds.
        self.version += 1

        total_metrics = aggregate_step_metrics(per_row_metrics)
        # Skip rate is a training-health signal: transitions dropped for
        # exceeding max_seq_len shrink the effective group and bias advantage
        # estimates toward short episodes. Alert if this climbs above ~10%.
        total_metrics["n_transitions"] = len(transitions)
        total_metrics["n_skipped_overlong"] = len(transitions) - B
        total_metrics["weights_version"] = self.version
        total_metrics.update(parity_stats)
        return total_metrics

    # -- weight sync -----------------------------------------------------------

    def push_adapter(self) -> str:
        """Save the current weights as policy_v{version} and hot-load them
        into vLLM. Returns the adapter name the proxy should route to.

        served_version advances only after vLLM confirms the load; on failure
        the trainer keeps training on newer weights while rollouts stay
        tagged with the old served version, and the recompute path treats
        those rows as off-policy (see _recompute_behavior_logprobs).
        "Already loaded" responses are retried with load_inplace=True
        (see _load_adapter_into_vllm).
        """
        import httpx

        name = f"policy_v{self.version}"
        if self.served_version == self.version:
            return name  # weights haven't moved since the last successful push

        prev = f"policy_v{self.served_version}" if self.served_version > 0 else None
        path = os.path.abspath(os.path.join(self.cfg.adapter_dir, name))
        self.model.save_pretrained(path)
        self._load_adapter_into_vllm(name, path)   # raises on real failure
        self.served_version = self.version
        log.info("loaded %s into vLLM", name)

        # Unload the superseded adapter so registrations don't accumulate
        # across a long run (30+ adapters risks hitting vLLM's LoRA slot
        # limits mid-run, which would strand training on a stale policy).
        # Best-effort: the proxy already routes new requests to the new name;
        # an in-flight request on the old adapter may fail and be retried by
        # the driver's rollout hygiene.
        if prev:
            try:
                httpx.post(
                    f"{self.cfg.vllm_base_url}/v1/unload_lora_adapter",
                    json={"lora_name": prev},
                    timeout=30,
                ).raise_for_status()
                log.info("unloaded %s", prev)
            except Exception as exc:
                log.warning("could not unload %s (continuing): %s", prev, exc)
        return name
