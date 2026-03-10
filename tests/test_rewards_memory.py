"""Tests for the memory invocation verifier."""

from __future__ import annotations

import pytest

from cardio.trainer.rewards.memory_penalty import MemoryInvocationVerifier


@pytest.fixture()
def verifier() -> MemoryInvocationVerifier:
    return MemoryInvocationVerifier(context_window=50)


def _make_token_offsets(text: str, n_tokens: int) -> list[int]:
    """Approximate character-level token offsets for testing."""
    chars_per_token = max(len(text) // n_tokens, 1)
    return [i * chars_per_token for i in range(n_tokens)]


class TestFindClinicalClaims:
    def test_find_clinical_claims_motion(self, verifier: MemoryInvocationVerifier) -> None:
        text = "The anterior wall shows hypokinesis. There is also akinesis in the septum."
        offsets = _make_token_offsets(text, 50)
        claims = verifier.find_clinical_claims(text, offsets)
        motion_claims = [c for c in claims if c["type"] == "motion"]
        assert len(motion_claims) == 2
        terms = {c["term"].lower() for c in motion_claims}
        assert "hypokinesis" in terms
        assert "akinesis" in terms

    def test_find_clinical_claims_structure(self, verifier: MemoryInvocationVerifier) -> None:
        text = "There is LV dilation and significant hypertrophy of the septum."
        offsets = _make_token_offsets(text, 50)
        claims = verifier.find_clinical_claims(text, offsets)
        structure_claims = [c for c in claims if c["type"] == "structure"]
        assert len(structure_claims) == 2
        terms = {c["term"].lower() for c in structure_claims}
        assert "dilation" in terms
        assert "hypertrophy" in terms


class TestVerifyInvocations:
    def test_verify_invocations_all_covered(self, verifier: MemoryInvocationVerifier) -> None:
        claims = [
            {"term": "hypokinesis", "type": "motion", "token_idx": 15, "sentence": "..."},
            {"term": "dilation", "type": "structure", "token_idx": 35, "sentence": "..."},
        ]
        invocation_log = [
            {"step": 10, "type": "tdm"},
            {"step": 30, "type": "psm"},
        ]
        penalty, violations = verifier.verify_invocations(claims, invocation_log)
        assert penalty == 0.0
        assert len(violations) == 0

    def test_verify_invocations_violation(self, verifier: MemoryInvocationVerifier) -> None:
        claims = [
            {"term": "hypokinesis", "type": "motion", "token_idx": 200, "sentence": "..."},
        ]
        invocation_log = [
            {"step": 10, "type": "tdm"},
        ]
        penalty, violations = verifier.verify_invocations(claims, invocation_log)
        assert penalty < 0.0
        assert len(violations) == 1

    def test_no_claims_no_penalty(self, verifier: MemoryInvocationVerifier) -> None:
        text = "The patient appears healthy with no notable findings."
        offsets = _make_token_offsets(text, 30)
        claims = verifier.find_clinical_claims(text, offsets)
        penalty, violations = verifier.verify_invocations(claims, [])
        assert penalty == 0.0
        assert len(violations) == 0
