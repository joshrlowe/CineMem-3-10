"""Tests for DCR claim isolation, IoU computation, and AutoMetricConverter."""

from __future__ import annotations

import numpy as np
import pytest

from cardio.trainer.rewards.dcr import AutoMetricConverter, DivideConquerEvaluator


@pytest.fixture()
def evaluator() -> DivideConquerEvaluator:
    return DivideConquerEvaluator()


class TestIsolateClaims:
    def test_isolate_claims_anatomy_pattern(self, evaluator: DivideConquerEvaluator) -> None:
        text = "The septal wall demonstrates hypokinesis."
        claims = evaluator.isolate_claims(text)
        assert len(claims) >= 1
        claim = claims[0]
        assert claim["anatomy"] == "septum"
        assert "hypokinesis" in claim["finding"]

    def test_isolate_claims_chamber_pattern(self, evaluator: DivideConquerEvaluator) -> None:
        text = "The LV is dilated."
        claims = evaluator.isolate_claims(text)
        assert len(claims) >= 1
        assert claims[0]["anatomy"].lower() == "lv"


class TestBboxIoU:
    def test_bbox_iou_perfect_overlap(self, evaluator: DivideConquerEvaluator) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 20:60] = 1
        pred_bbox = [20.0, 10.0, 60.0, 50.0]
        iou = evaluator.compute_iou(pred_bbox, mask)
        assert iou == pytest.approx(1.0)

    def test_bbox_iou_no_overlap(self, evaluator: DivideConquerEvaluator) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 1
        pred_bbox = [80.0, 80.0, 99.0, 99.0]
        iou = evaluator.compute_iou(pred_bbox, mask)
        assert iou == pytest.approx(0.0)

    def test_bbox_iou_partial(self, evaluator: DivideConquerEvaluator) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:50, 20:60] = 1
        pred_bbox = [30.0, 20.0, 70.0, 60.0]
        iou = evaluator.compute_iou(pred_bbox, mask)
        assert 0.0 < iou < 1.0


class TestEvaluateClaims:
    def test_evaluate_claims_grounding_status(self, evaluator: DivideConquerEvaluator) -> None:
        gt_mask = np.zeros((100, 100), dtype=np.uint8)
        gt_mask[10:50, 20:60] = 1

        claims = [
            {"sentence": "test", "anatomy": "lv", "finding": "dilated",
             "start_char": 0, "end_char": 10},
        ]

        perfect_bbox = {"lv": [20.0, 10.0, 60.0, 50.0]}
        results = evaluator.evaluate_claims(claims, perfect_bbox, {"lv": gt_mask})
        assert results[0]["grounding_status"] == "correct"
        assert results[0]["iou"] >= 0.5

        bad_bbox = {"lv": [80.0, 80.0, 99.0, 99.0]}
        results = evaluator.evaluate_claims(claims, bad_bbox, {"lv": gt_mask})
        assert results[0]["grounding_status"] == "hallucination"
        assert results[0]["iou"] < 0.2


class TestAutoMetricConverter:
    def test_compute_penalty_correct(self) -> None:
        assert AutoMetricConverter.compute_penalty(0.6) == pytest.approx(0.0)
        assert AutoMetricConverter.compute_penalty(0.5) == pytest.approx(0.0)

    def test_compute_penalty_hallucination(self) -> None:
        assert AutoMetricConverter.compute_penalty(0.1) == pytest.approx(1.0)
        assert AutoMetricConverter.compute_penalty(0.0) == pytest.approx(1.0)

    def test_compute_penalty_partial(self) -> None:
        penalty = AutoMetricConverter.compute_penalty(0.35)
        assert 0.0 < penalty < 1.0
        assert penalty == pytest.approx(1.0 - 0.35 / 0.5)
