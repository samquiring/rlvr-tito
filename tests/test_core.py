"""Tests for the parts where silent bugs live: advantage math, ratio/clip
behavior, collation alignment (the mask/logprob/token correspondence), and
group bookkeeping. Run: python -m pytest tests/ -q  (or plain python)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from rlvr_tito.grpo import group_advantages, grpo_loss, gather_logprobs
from rlvr_tito.store import GroupStore, Transition
from rlvr_tito.trainer import collate_transitions, flatten_group, _check_parity_match


def test_group_advantages():
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(adv.sum(), torch.tensor(0.0), atol=1e-6)
    assert adv[0] > 0 and adv[1] < 0
    # degenerate group -> all zeros
    assert group_advantages(torch.zeros(8)).abs().sum() == 0


def test_grpo_loss_direction_and_masking():
    """Positive advantage must push logp up; masked positions contribute 0."""
    B, T = 2, 6
    logp_new = torch.full((B, T), -2.0, requires_grad=True)
    logp_beh = torch.full((B, T), -2.0)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 1, 1, 1]], dtype=torch.float32)

    loss, m = grpo_loss(logp_new, logp_beh, adv, mask)
    loss.backward()
    g = logp_new.grad
    # gradient of loss wrt logp: negative where adv>0 (increase logp),
    # positive where adv<0, exactly zero on masked (prompt) positions
    assert (g[0][mask[0] == 1] < 0).all()
    assert (g[1][mask[1] == 1] > 0).all()
    assert (g[mask == 0] == 0).all()
    assert m["response_tokens"] == 7.0


def test_grpo_clip_stops_gradient():
    """Once the ratio exceeds the clip and the advantage is positive, the
    objective must go flat (no further gradient)."""
    logp_beh = torch.tensor([[-2.0]])
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    # ratio = exp(0.5) ~ 1.65 > 1 + clip_high
    logp_new = torch.tensor([[-1.5]], requires_grad=True)
    loss, _ = grpo_loss(logp_new, logp_beh, adv, mask,
                        clip_low=0.2, clip_high=0.28)
    loss.backward()
    assert torch.allclose(logp_new.grad, torch.zeros_like(logp_new.grad))


def test_gather_logprobs_shape_and_value():
    logits = torch.zeros(1, 3, 5)          # uniform -> logp = -log(5)
    labels = torch.tensor([[0, 3, 4]])
    lp = gather_logprobs(logits, labels)
    assert lp.shape == (1, 3)
    assert torch.allclose(lp, torch.full((1, 3), -torch.log(torch.tensor(5.0))))


def test_collation_alignment():
    """The invariant everything depends on: at every padded position, mask,
    behavior logprob, and token id must refer to the same token."""
    t1 = Transition([10, 11, 12], [20, 21], [-0.5, -0.7], adapter_version=0)
    t2 = Transition([10], [30, 31, 32], [-0.1, -0.2, -0.3], adapter_version=0)
    batch = collate_transitions([t1, t2], [1.0, -1.0],
                                pad_token_id=0, max_seq_len=64)
    ids, mask, blp = (batch["input_ids"], batch["response_mask"],
                      batch["behavior_logprobs"])
    assert ids.shape == mask.shape == blp.shape == (2, 5)
    # row 0: prompt(3) + response(2); row 1: prompt(1) + response(3) + pad(1)
    assert ids[0].tolist() == [10, 11, 12, 20, 21]
    assert mask[0].tolist() == [0, 0, 0, 1, 1]
    assert torch.allclose(blp[0], torch.tensor([0, 0, 0, -0.5, -0.7]))
    assert ids[1].tolist() == [10, 30, 31, 32, 0]        # padded
    assert mask[1].tolist() == [0, 1, 1, 1, 0]
    assert torch.allclose(blp[1], torch.tensor([0, -0.1, -0.2, -0.3, 0]))


def test_collation_left_truncates_prompt_only():
    tr = Transition(list(range(100)), [7, 8, 9], [-1.0, -1.0, -1.0], 0)
    batch = collate_transitions([tr], [1.0], pad_token_id=0, max_seq_len=10)
    ids, mask = batch["input_ids"], batch["response_mask"]
    assert ids.shape[1] == 10
    assert ids[0][-3:].tolist() == [7, 8, 9]       # response intact
    assert mask[0][-3:].tolist() == [1, 1, 1]
    assert mask[0][:-3].sum() == 0


def test_transition_rejects_misaligned_logprobs():
    try:
        Transition([1], [2, 3], [-0.1], adapter_version=0)
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_group_store_lifecycle():
    s = GroupStore(group_size=2)
    for task, reward in [("A", 1.0), ("A", 0.0), ("B", 1.0)]:
        tid = s.start_trajectory(task)
        s.record(tid, Transition([1], [2], [-0.1], 0))
        s.complete(tid, reward)
    group = s.pop_ready_group()
    assert group is not None and len(group) == 2
    assert {t.task_id for t in group} == {"A"}
    assert s.pop_ready_group() is None                  # B incomplete


def test_group_store_drops_degenerate():
    s = GroupStore(group_size=2, drop_degenerate=True)
    for _ in range(2):
        tid = s.start_trajectory("A")
        s.record(tid, Transition([1], [2], [-0.1], 0))
        s.complete(tid, 0.0)                            # all-fail group
    assert s.pop_ready_group() is None
    assert s.stats["groups_dropped_degenerate"] == 1


def test_flatten_group_broadcasts_advantage():
    from rlvr_tito.store import Trajectory
    t_win = Trajectory("w", "A", [Transition([1], [2], [-0.1], 0)] * 3, reward=1.0)
    t_lose = Trajectory("l", "A", [Transition([1], [2], [-0.1], 0)] * 2, reward=0.0)
    transitions, advs = flatten_group([t_win, t_lose])
    assert len(transitions) == 5
    assert advs[0] == advs[1] == advs[2] > 0
    assert advs[3] == advs[4] < 0


def test_microbatch_gradient_equals_full_batch():
    """Micro-batch accumulated gradient must equal the single-pass full-batch gradient.

    Failure mode caught here: scaling a local token-mean by (local_samples / B)
    makes the effective per-token weight vary with micro-batch token count, so
    the accumulated gradient is NOT a token-mean over the full batch.
    The fix scales each micro-batch by (local_tokens / global_tokens) instead.
    """
    torch.manual_seed(42)
    B, T = 6, 8
    logp_beh = torch.randn(B, T)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.8, -0.3])
    # Unequal token counts per row to make the bug visible
    mask = torch.zeros(B, T)
    mask[0, :7] = 1  # 7 tokens
    mask[1, :2] = 1  # 2 tokens
    mask[2, :5] = 1
    mask[3, :4] = 1
    mask[4, :6] = 1
    mask[5, :1] = 1  # 1 token

    # -- full-batch reference gradient --
    logp_full = torch.randn(B, T, requires_grad=True)
    loss_full, _ = grpo_loss(logp_full, logp_beh, adv, mask)
    loss_full.backward()
    grad_ref = logp_full.grad.clone()

    # -- micro-batch accumulation with the corrected scaling --
    logp_mb = logp_full.detach().clone().requires_grad_(True)
    global_tokens = float(mask.sum().clamp(min=1))
    mb = 2
    for i in range(0, B, mb):
        sl = slice(i, i + mb)
        loss_mb, _ = grpo_loss(logp_mb[sl], logp_beh[sl], adv[sl], mask[sl])
        local_tokens = float(mask[sl].sum().clamp(min=1))
        (loss_mb * (local_tokens / global_tokens)).backward()

    assert torch.allclose(grad_ref, logp_mb.grad, atol=1e-6), (
        f"max diff {(grad_ref - logp_mb.grad).abs().max():.2e}"
    )


def test_parity_check_passes_within_tolerance():
    _check_parity_match([-1.01, -2.005], [-1.0, -2.0], tol=0.05, idx=0)


def test_parity_check_detects_logprob_drift():
    try:
        _check_parity_match([-1.0, -2.0], [-1.5, -2.5], tol=0.05, idx=0)
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "drift" in str(e).lower() or "drifted" in str(e).lower()


def test_parity_check_detects_length_mismatch():
    try:
        _check_parity_match([-1.0, -2.0], [-1.0], tol=0.05, idx=1)
        raise AssertionError("should have raised")
    except RuntimeError as e:
        assert "length mismatch" in str(e).lower()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
