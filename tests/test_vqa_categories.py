"""Tests for VQA category definitions and answer logic."""

from __future__ import annotations

from cardio.data.vqa import (
    ALL_CATEGORIES,
    CAT_ABNORMALITY,
    MULTI_VIEW_INPUTS,
    VQACategory,
)


class TestVQACategoryEnum:
    def test_vqa_category_enum_count(self) -> None:
        members = list(VQACategory)
        assert len(members) == 19

    def test_enum_is_string(self) -> None:
        assert VQACategory.LV_FUNCTION == "lv_systolic_function"
        assert isinstance(VQACategory.LV_FUNCTION, str)


class TestCategoryAliases:
    def test_all_categories_list(self) -> None:
        assert len(ALL_CATEGORIES) == 7
        for cat in ALL_CATEGORIES:
            assert isinstance(cat, str)

    def test_category_string_aliases(self) -> None:
        assert CAT_ABNORMALITY == "abnormality_localization"


class TestMultiViewInputs:
    def test_multi_view_inputs_keys(self) -> None:
        assert len(MULTI_VIEW_INPUTS) == 6
        expected_keys = {
            VQACategory.CROSS_VIEW_STRUCTURE,
            VQACategory.MULTI_VIEW_CONTRACTION,
            VQACategory.RV_ASSESSMENT,
            VQACategory.MULTI_PHASE_SUMMARY,
            VQACategory.CROSS_VIEW_CONSISTENCY,
            VQACategory.TEMPORAL_CONTRACTION,
        }
        assert set(MULTI_VIEW_INPUTS.keys()) == expected_keys
