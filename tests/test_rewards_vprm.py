"""Tests for the ACC/AHA deterministic verifier (VPRM)."""

from __future__ import annotations

import pytest

from cardio.trainer.rewards.vprm import ACCAHAVerifier, _LVEF_RE


@pytest.fixture()
def verifier() -> ACCAHAVerifier:
    return ACCAHAVerifier()


class TestHFpEF:
    def test_hfpef_full_evidence(self, verifier: ACCAHAVerifier) -> None:
        text = (
            "LVEF is 55%. LAVI of 38 mL/m2 indicates left atrial enlargement. "
            "The diagnosis is HFpEF."
        )
        reward, violations = verifier.verify_hf_classification(
            text, "HFpEF", "HFpEF"
        )
        assert reward == 1.0
        assert not violations

    def test_hfpef_no_filling_pressure(self, verifier: ACCAHAVerifier) -> None:
        text = "LVEF is 55%. The diagnosis is HFpEF."
        reward, violations = verifier.verify_hf_classification(
            text, "HFpEF", "HFpEF"
        )
        assert reward == 0.5
        assert len(violations) >= 1

    def test_hfpef_contradictory_ef(self, verifier: ACCAHAVerifier) -> None:
        text = "LVEF is 35%. The diagnosis is HFpEF."
        reward, violations = verifier.verify_hf_classification(
            text, "HFpEF", "HFpEF"
        )
        assert reward == 0.0
        assert any("LVEF" in v for v in violations)


class TestHFrEF:
    def test_hfref_correct(self, verifier: ACCAHAVerifier) -> None:
        text = "LVEF is 30%. Severely reduced systolic function. HFrEF."
        reward, violations = verifier.verify_hf_classification(
            text, "HFrEF", "HFrEF"
        )
        assert reward == 1.0
        assert not violations

    def test_hfref_ef_too_high(self, verifier: ACCAHAVerifier) -> None:
        text = "LVEF is 55%. The diagnosis is HFrEF."
        reward, violations = verifier.verify_hf_classification(
            text, "HFrEF", "HFrEF"
        )
        assert reward == 0.0
        assert len(violations) >= 1


class TestRegexExtraction:
    def test_regex_formats(self) -> None:
        patterns = [
            "LVEF is 45%",
            "LVEF of 45%",
            "LVEF: 45%",
            "LVEF = 45%",
            "LVEF 45%",
        ]
        for text in patterns:
            m = _LVEF_RE.search(text)
            assert m is not None, f"Failed to match: {text!r}"
            assert float(m.group(1)) == pytest.approx(45.0)

    def test_extract_metrics_multiple(self, verifier: ACCAHAVerifier) -> None:
        text = "LVEF is 50%. LAVI of 36 mL/m2. GLS of -17%."
        metrics = verifier.extract_metrics(text)
        assert metrics["lvef"] == pytest.approx(50.0)
        assert metrics["lavi"] == pytest.approx(36.0)
        assert metrics["gls"] == pytest.approx(-17.0)


class TestEFConsistency:
    def test_ef_consistency_mismatch(self, verifier: ACCAHAVerifier) -> None:
        text = (
            "LVEF is 60%. The patient demonstrates severely reduced contraction."
        )
        reward, violations = verifier.verify_ef_consistency(text)
        assert reward == 0.0
        assert len(violations) >= 1
        assert any("mismatch" in v.lower() for v in violations)
