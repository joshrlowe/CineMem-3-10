"""Reward functions, trainers, and optimization utilities for CardioVLM."""

from cardio.trainer.finetune import maybe_reduce_batch_size, maybe_subset_dataset, run_train
from cardio.trainer.optim import (
    CosineScheduler,
    EarlyStopping,
    GradScaler,
    WarmupThenCosineScheduler,
    adjust_learning_rate,
    clip_grad_norm,
    get_n_accum_steps,
)
from cardio.trainer.rewards.cardiac import cardiac_reward, cardiac_reward_normalised
from cardio.trainer.rewards.composite import CompositeRewardEngine
from cardio.trainer.rewards.dcr import AutoMetricConverter, DivideConquerEvaluator
from cardio.trainer.rewards.exact_match import exact_match_reward, substring_reward
from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier
from cardio.trainer.rewards.vprm import ACCAHAVerifier
from cardio.trainer.stage1_memory import stage1_loss

__all__ = [
    "ACCAHAVerifier",
    "AutoMetricConverter",
    "CompositeRewardEngine",
    "CosineScheduler",
    "DivideConquerEvaluator",
    "EarlyStopping",
    "GradScaler",
    "MemoryInvocationVerifier",
    "WarmupThenCosineScheduler",
    "adjust_learning_rate",
    "cardiac_reward",
    "cardiac_reward_normalised",
    "clip_grad_norm",
    "exact_match_reward",
    "get_n_accum_steps",
    "maybe_reduce_batch_size",
    "maybe_subset_dataset",
    "run_train",
    "stage1_loss",
    "substring_reward",
]
