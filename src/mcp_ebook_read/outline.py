"""Pure helpers for outline resolution and node-scoped matching."""

from __future__ import annotations

from dataclasses import dataclass

from mcp_ebook_read.schema.models import OutlineNode


@dataclass(frozen=True)
class ResolvedOutlineNode:
    node: OutlineNode
    path: list[str]


def page_ranges_overlap(
    chunk_range: list[int] | None,
    node_start: int | None,
    node_end: int | None,
) -> bool:
    if not chunk_range or len(chunk_range) != 2:
        return False
    if node_start is None or node_end is None:
        return False
    c_start, c_end = chunk_range
    return c_start <= node_end and c_end >= node_start


def normalize_section_key(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def normalize_section_path(section_path: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in section_path:
        item = normalize_section_key(part)
        if item:
            normalized.append(item)
    return normalized


def section_path_prefix_matches(section_path: list[str], node_path: list[str]) -> bool:
    normalized_section = normalize_section_path(section_path)
    normalized_node = normalize_section_path(node_path)
    if not normalized_section or not normalized_node:
        return False
    if len(normalized_section) < len(normalized_node):
        return False
    return normalized_section[: len(normalized_node)] == normalized_node


def section_path_leaf_matches(section_path: list[str], node_path: list[str]) -> bool:
    normalized_section = normalize_section_path(section_path)
    normalized_node = normalize_section_path(node_path)
    if not normalized_section or not normalized_node:
        return False
    return normalized_section[-1] == normalized_node[-1]


def find_outline_node(
    nodes: list[OutlineNode],
    node_id: str,
    ancestry: list[str] | None = None,
) -> ResolvedOutlineNode | None:
    ancestry = ancestry or []
    for node in nodes:
        node_path = ancestry + [node.title]
        if node.id == node_id:
            return ResolvedOutlineNode(node=node, path=node_path)
        matched = find_outline_node(node.children, node_id, node_path)
        if matched is not None:
            return matched
    return None


def matches_outline_node(
    *,
    page_range: list[int] | None,
    section_path: list[str],
    spine_id: str | None,
    node: OutlineNode,
    node_path: list[str],
) -> bool:
    if page_ranges_overlap(page_range, node.page_start, node.page_end):
        return True

    if node.spine_ref and spine_id:
        if spine_id != node.spine_ref:
            return False
        if section_path_prefix_matches(section_path, node_path):
            return True

    if section_path_prefix_matches(section_path, node_path):
        return True
    return section_path_leaf_matches(section_path, node_path)
