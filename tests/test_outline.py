from __future__ import annotations

from mcp_ebook_read.outline import (
    find_outline_node,
    matches_outline_node,
    normalize_section_key,
    normalize_section_path,
    page_ranges_overlap,
    section_path_leaf_matches,
    section_path_prefix_matches,
)
from mcp_ebook_read.schema.models import OutlineNode


def test_page_ranges_overlap_handles_true_and_false_cases() -> None:
    assert page_ranges_overlap([2, 4], 3, 5) is True
    assert page_ranges_overlap([1], 1, 1) is False
    assert page_ranges_overlap([1, 2], None, 2) is False
    assert page_ranges_overlap([1, 2], 3, None) is False


def test_section_path_normalization_and_matching_helpers() -> None:
    assert normalize_section_key("  Chapter   One ") == "chapter one"
    assert normalize_section_path([" Chapter One ", "", "Summary"]) == [
        "chapter one",
        "summary",
    ]
    assert (
        section_path_prefix_matches(
            ["Chapter One", "Summary"],
            ["chapter one"],
        )
        is True
    )
    assert section_path_prefix_matches(["Appendix"], ["Chapter One"]) is False
    assert section_path_prefix_matches([], ["Chapter One"]) is False
    assert (
        section_path_prefix_matches(["Chapter One"], ["Chapter One", "Summary"])
        is False
    )
    assert (
        section_path_leaf_matches(
            ["Chapter One", "Summary"],
            ["Chapter Two", "summary"],
        )
        is True
    )
    assert section_path_leaf_matches(["Chapter One"], ["Appendix"]) is False
    assert section_path_leaf_matches([], ["Appendix"]) is False


def test_find_outline_node_supports_nested_children() -> None:
    outline = [
        OutlineNode(
            id="parent",
            title="Chapter One",
            level=1,
            children=[
                OutlineNode(
                    id="child",
                    title="Section 1.1",
                    level=2,
                )
            ],
        )
    ]

    resolved = find_outline_node(outline, "child")
    assert resolved is not None
    assert resolved.node.id == "child"
    assert resolved.path == ["Chapter One", "Section 1.1"]
    assert find_outline_node(outline, "missing") is None


def test_matches_outline_node_uses_page_range_first() -> None:
    node = OutlineNode(
        id="page-node", title="Method", level=1, page_start=3, page_end=5
    )
    assert (
        matches_outline_node(
            page_range=[4, 4],
            section_path=["Ignored"],
            spine_id=None,
            node=node,
            node_path=["Method"],
        )
        is True
    )


def test_matches_outline_node_rejects_wrong_spine() -> None:
    node = OutlineNode(id="epub-node", title="Summary", level=2, spine_ref="chap1")
    assert (
        matches_outline_node(
            page_range=None,
            section_path=["Chapter One", "Summary"],
            spine_id="chap2",
            node=node,
            node_path=["Chapter One", "Summary"],
        )
        is False
    )


def test_matches_outline_node_accepts_spine_and_prefix_path() -> None:
    node = OutlineNode(id="epub-node", title="Summary", level=2, spine_ref="chap1")
    assert (
        matches_outline_node(
            page_range=None,
            section_path=["Chapter One", "Summary"],
            spine_id="chap1",
            node=node,
            node_path=["Chapter One", "Summary"],
        )
        is True
    )


def test_matches_outline_node_accepts_prefix_without_spine() -> None:
    node = OutlineNode(id="prefix-node", title="Chapter One", level=1)
    assert (
        matches_outline_node(
            page_range=None,
            section_path=["Chapter One", "Section 1.1"],
            spine_id=None,
            node=node,
            node_path=["Chapter One"],
        )
        is True
    )


def test_matches_outline_node_falls_back_to_leaf_title_equality() -> None:
    node = OutlineNode(id="leaf-node", title="Summary", level=2)
    assert (
        matches_outline_node(
            page_range=None,
            section_path=["Appendix", "Summary"],
            spine_id=None,
            node=node,
            node_path=["Chapter One", "Summary"],
        )
        is True
    )


def test_matches_outline_node_returns_false_when_no_strategy_matches() -> None:
    node = OutlineNode(id="miss-node", title="Method", level=1)
    assert (
        matches_outline_node(
            page_range=None,
            section_path=["Appendix", "Results"],
            spine_id=None,
            node=node,
            node_path=["Method"],
        )
        is False
    )
