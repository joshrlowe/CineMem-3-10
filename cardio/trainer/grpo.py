"""GRPO and NGRPO trainers for cardiac VQA.

Ports the reference ``SimpleGRPOTrainer`` from CineMem with bug fixes,
then extends it with :class:`NGRPOTrainer` for calibrated advantages,
asymmetric clipping, and TR-GRPO token-level loss modulation.

Bug fixes vs. reference
-----------------------
1. ``reward_ema`` is an **instance variable** instead of a fragile
   module-level global that required ``global`` declarations.
2. :meth:`GRPOTrainer.compute_advantages` uses full normalisation
   ``(r - mean) / (std + eps)`` instead of mean-only subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

# =========================================================================
# Module-level utilities
# =========================================================================


@dataclass
class GRPOBatch:
    """Tokenised batch for GRPO training."""

    input_ids: torch.LongTensor  # (B, T)
    attention_mask: torch.LongTensor  # (B, T)
    labels: torch.LongTensor  # (B, T) — prompt positions set to -100


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> torch.Tensor:
    """Sum of per-token log-probs for non-masked positions.

    Args:
        logits: ``(B, T, V)`` raw logits from the model.
        labels: ``(B, T)`` token ids with ``-100`` for positions to ignore.

    Returns:
        ``(B,)`` sequence-level log-probability sums.
    """
    logp = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * mask
    return gathered.sum(dim=-1)


def per_token_logprobs(
    logits: torch.Tensor,
    labels: torch.LongTensor,
) -> torch.Tensor:
    """Per-token log-probs, zeroed at masked positions.

    Same logic as :func:`sequence_logprobs` but without the final sum,
    returning ``(B, T)`` for token-level loss modulation in
    :class:`NGRPOTrainer`.
    """
    logp = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    gathered = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return gathered * mask


def kl_divergence(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence KL divergence ``KL(p || q)``.

    Args:
        logits_p: ``(B, T, V)`` logits of the current policy.
        logits_q: ``(B, T, V)`` logits of the reference policy.

    Returns:
        ``(B,)`` mean-over-time KL for each sequence.
    """
    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)
    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)  # (B, T)
    return kl.mean(dim=-1)


# =========================================================================
# GRPOTrainer (base)
# =========================================================================


class GRPOTrainer:
    """Base Group Relative Policy Optimization.

    Ported from the CineMem reference with two fixes:

    1. ``reward_ema`` is an instance variable, not a module-level global.
    2. ``compute_advantages`` uses full ``(r - mean) / (std + eps)``
       normalisation.

    .. warning::
       If all rewards in a group are identical, ``std == 0`` and
       advantages collapse to zero, producing no gradient signal.
       Use :class:`NGRPOTrainer` to resolve this deadlock.
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any | None = None,
        kl_beta: float = 0.02,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.kl_beta = kl_beta
        self.reward_ema: float = 0.0
        self.ema_alpha: float = 0.05

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """Standard GRPO advantage normalisation.

        ``advantages = (rewards - mean) / (std + eps)``

        Returns all-zeros when ``std == 0`` (homogeneous group).
        """
        mean = rewards.mean()
        std = rewards.std()
        if std < 1e-8:
            return torch.zeros_like(rewards)
        return (rewards - mean) / (std + 1e-8)

    # ------------------------------------------------------------------
    # Reward EMA tracking
    # ------------------------------------------------------------------

    def update_reward_ema(self, reward: float) -> None:
        """Exponential moving average update for reward baseline."""
        self.reward_ema = (
            (1.0 - self.ema_alpha) * self.reward_ema
            + self.ema_alpha * reward
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def loss_from_samples(
        self,
        prompts_inputs: dict[str, Any],
        sampled_ids: torch.LongTensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GRPO loss from prompt inputs, sampled token ids, and rewards.

        Concatenates prompt and sampled ids, runs a forward pass, and
        returns the policy-gradient loss with an optional KL penalty.
        """
        input_ids = torch.cat(
            [prompts_inputs["input_ids"], sampled_ids], dim=1
        )
        attn = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, : prompts_inputs["input_ids"].size(1)] = -100

        out = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=False,
        )
        logits = out.logits
        logp = sequence_logprobs(logits[:, :-1, :], labels[:, 1:])

        adv = self.compute_advantages(rewards)
        pg_loss = -(adv.detach() * logp).mean()

        if self.ref_model is None or self.kl_beta <= 0:
            return pg_loss

        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attn)
        kl = kl_divergence(logits[:, :-1, :], ref_out.logits[:, :-1, :])
        return pg_loss + self.kl_beta * kl.mean()


# =========================================================================
# NGRPOTrainer
# =========================================================================


class NGRPOTrainer(GRPOTrainer):
    """Negative-enhanced GRPO with Advantage Calibration.

    Resolves the zero-gradient deadlock of standard GRPO by injecting a
    virtual maximum-reward sample (``r_max``) into the advantage
    calculation, and applies asymmetric PPO-style clipping with
    token-level penalty modulation for TR-GRPO.

    Args:
        r_max: virtual maximum reward appended for calibration.
        epsilon_neg: clip range for negative advantages (tighter).
        epsilon_pos: clip range for positive advantages (wider).
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any | None = None,
        kl_beta: float = 0.02,
        r_max: float = 1.0,
        epsilon_neg: float = 0.16,
        epsilon_pos: float = 0.24,
    ) -> None:
        super().__init__(model, ref_model, kl_beta)
        self.r_max = r_max
        self.epsilon_neg = epsilon_neg
        self.epsilon_pos = epsilon_pos

    # ------------------------------------------------------------------
    # Calibrated advantages
    # ------------------------------------------------------------------

    def compute_calibrated_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """NGRPO advantage calibration with virtual ``r_max`` injection.

        Appends ``r_max`` as a G+1 virtual sample, computes
        ``calibrated_mean`` and ``calibrated_std`` over all G+1 values,
        then normalises only the original G rewards.  This guarantees
        ``std > 0`` even when all actual rewards are identical.
        """
        r_max_t = torch.tensor(
            [self.r_max], dtype=rewards.dtype, device=rewards.device
        )
        augmented = torch.cat([rewards, r_max_t])

        cal_mean = augmented.mean()
        cal_std = augmented.std()

        return (rewards - cal_mean) / (cal_std + 1e-8)

    # ------------------------------------------------------------------
    # Asymmetric clipping
    # ------------------------------------------------------------------

    def asymmetric_clip(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """PPO-style clipping with different ranges for +/- advantages.

        Positive advantages use ``[1 - eps_pos, 1 + eps_pos]`` (wider
        range, allowing strong positive learning).  Negative advantages
        use ``[1 - eps_neg, 1 + eps_neg]`` (tighter range, preventing
        catastrophic forgetting).
        """
        pos_mask = advantages >= 0

        clipped = torch.empty_like(ratio)
        clipped[pos_mask] = torch.clamp(
            ratio[pos_mask],
            1.0 - self.epsilon_pos,
            1.0 + self.epsilon_pos,
        )
        clipped[~pos_mask] = torch.clamp(
            ratio[~pos_mask],
            1.0 - self.epsilon_neg,
            1.0 + self.epsilon_neg,
        )
        return clipped

    # ------------------------------------------------------------------
    # Token-level loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        token_penalty_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """TR-NGRPO per-token surrogate loss.

        Args:
            old_logprobs: ``(B, T)`` per-token log-probs under the old
                policy (detached).
            new_logprobs: ``(B, T)`` per-token log-probs under the
                current policy.
            advantages: ``(B,)`` calibrated advantages.
            token_penalty_mask: optional ``(T,)`` or ``(B, T)`` penalty
                mask from DCR/AMC.  Values in ``[0, 1]``; tokens with
                higher penalty receive amplified loss.

        Returns:
            Scalar mean negative surrogate loss.
        """
        ratio = torch.exp(new_logprobs - old_logprobs)  # (B, T)

        adv_expanded = advantages.unsqueeze(-1).expand_as(ratio)  # (B, T)

        clipped_ratio = self.asymmetric_clip(ratio, adv_expanded)

        surr1 = ratio * adv_expanded
        surr2 = clipped_ratio * adv_expanded
        surrogate = torch.min(surr1, surr2)  # (B, T)

        if token_penalty_mask is not None:
            if token_penalty_mask.dim() == 1:
                token_penalty_mask = token_penalty_mask.unsqueeze(0)
            penalty_len = token_penalty_mask.size(-1)
            surr_len = surrogate.size(-1)
            if penalty_len < surr_len:
                pad = torch.zeros(
                    *token_penalty_mask.shape[:-1],
                    surr_len - penalty_len,
                    dtype=token_penalty_mask.dtype,
                    device=token_penalty_mask.device,
                )
                token_penalty_mask = torch.cat(
                    [token_penalty_mask, pad], dim=-1
                )
            elif penalty_len > surr_len:
                token_penalty_mask = token_penalty_mask[..., :surr_len]
            surrogate = surrogate * (1.0 + token_penalty_mask)

        active = new_logprobs != 0
        if active.any():
            loss = -surrogate[active].mean()
        else:
            loss = -surrogate.mean()

        return loss

    # ------------------------------------------------------------------
    # Training step (override)
    # ------------------------------------------------------------------

    def loss_from_samples(  # type: ignore[override]
        self,
        prompts_inputs: dict[str, Any],
        sampled_ids: torch.LongTensor,
        rewards: torch.Tensor,
        token_penalty_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NGRPO training step with calibrated advantages and token-level loss.

        Same interface as :meth:`GRPOTrainer.loss_from_samples` with an
        additional *token_penalty_mask* argument from the DCR/AMC
        pipeline.
        """
        input_ids = torch.cat(
            [prompts_inputs["input_ids"], sampled_ids], dim=1
        )
        attn = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, : prompts_inputs["input_ids"].size(1)] = -100

        shifted_labels = labels[:, 1:]
        shifted_logits_slice = slice(None, -1)

        # Old (detached) log-probs
        with torch.no_grad():
            old_out = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=False,
            )
            old_lp = per_token_logprobs(
                old_out.logits[:, shifted_logits_slice, :], shifted_labels
            )

        # New log-probs (with gradient)
        new_out = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=False,
        )
        new_logits = new_out.logits
        new_lp = per_token_logprobs(
            new_logits[:, shifted_logits_slice, :], shifted_labels
        )

        adv = self.compute_calibrated_advantages(rewards)

        pg_loss = self.compute_loss(
            old_lp, new_lp, adv, token_penalty_mask
        )

        if self.ref_model is None or self.kl_beta <= 0:
            return pg_loss

        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids, attention_mask=attn)
        kl = kl_divergence(
            new_logits[:, shifted_logits_slice, :],
            ref_out.logits[:, shifted_logits_slice, :],
        )
        return pg_loss + self.kl_beta * kl.mean()
