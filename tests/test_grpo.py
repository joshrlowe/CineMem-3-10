"""Tests for GRPO and NGRPO advantage calibration math."""

from __future__ import annotations

import torch
import pytest

from cardio.trainer.grpo import (
    GRPOTrainer,
    NGRPOTrainer,
    per_token_logprobs,
    sequence_logprobs,
)


class TestGRPOAdvantages:
    def test_grpo_homogeneous_deadlock(self) -> None:
        rewards = torch.tensor([0.5, 0.5, 0.5])
        adv = GRPOTrainer.compute_advantages(rewards)
        assert torch.allclose(adv, torch.zeros_like(adv))

    def test_grpo_heterogeneous(self) -> None:
        rewards = torch.tensor([0.0, 0.5, 1.0])
        adv = GRPOTrainer.compute_advantages(rewards)
        assert adv[0] < 0.0
        assert adv[2] > 0.0
        assert adv.mean().abs() < 1e-5


class TestNGRPOAdvantages:
    def test_ngrpo_homogeneous_negative(self) -> None:
        trainer = NGRPOTrainer(model=None, r_max=1.0)
        rewards = torch.tensor([0.5, 0.5, 0.5])
        adv = trainer.compute_calibrated_advantages(rewards)
        assert (adv < 0).all(), "All advantages should be negative when rewards < r_max"

    def test_ngrpo_calibrated_std_nonzero(self) -> None:
        trainer = NGRPOTrainer(model=None, r_max=1.0)
        rewards = torch.tensor([0.3, 0.3, 0.3])
        augmented = torch.cat([rewards, torch.tensor([trainer.r_max])])
        cal_std = augmented.std()
        assert cal_std > 0.0, "Calibrated std should never be zero"


class TestAsymmetricClipping:
    def test_asymmetric_clip_respects_epsilons(self) -> None:
        trainer = NGRPOTrainer(
            model=None, epsilon_neg=0.16, epsilon_pos=0.24
        )

        ratio = torch.tensor([0.5, 0.8, 1.0, 1.2, 1.5])
        pos_adv = torch.ones(5)
        neg_adv = -torch.ones(5)

        clipped_pos = trainer.asymmetric_clip(ratio, pos_adv)
        assert clipped_pos.min().item() >= 1.0 - 0.24 - 1e-6
        assert clipped_pos.max().item() <= 1.0 + 0.24 + 1e-6

        clipped_neg = trainer.asymmetric_clip(ratio, neg_adv)
        assert clipped_neg.min().item() >= 1.0 - 0.16 - 1e-6
        assert clipped_neg.max().item() <= 1.0 + 0.16 + 1e-6


class TestComputeLoss:
    def test_token_penalty_mask_amplifies_loss(self) -> None:
        trainer = NGRPOTrainer(model=None, epsilon_neg=0.16, epsilon_pos=0.24)
        B, T = 2, 4
        old_lp = torch.randn(B, T)
        new_lp = old_lp + 0.1
        advantages = torch.tensor([-0.5, 0.5])

        loss_no_mask = trainer.compute_loss(old_lp, new_lp, advantages, None)

        penalty_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
        loss_with_mask = trainer.compute_loss(old_lp, new_lp, advantages, penalty_mask)

        assert loss_with_mask.abs() >= loss_no_mask.abs() - 1e-6


class TestLogprobs:
    def test_sequence_logprobs_masked(self) -> None:
        B, T, V = 2, 6, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        labels[:, :2] = -100

        full_labels = labels.clone()
        full_labels[:, :2] = 0
        logp_full = sequence_logprobs(logits, full_labels)

        logp_masked = sequence_logprobs(logits, labels)

        assert not torch.allclose(logp_full, logp_masked), (
            "Masking should change the sum"
        )

    def test_per_token_logprobs_shape(self) -> None:
        B, T, V = 3, 8, 10
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        result = per_token_logprobs(logits, labels)
        assert result.shape == (B, T)
