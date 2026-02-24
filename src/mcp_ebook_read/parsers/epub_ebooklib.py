"""EPUB parser using EbookLib + lxml."""

from __future__ import annotations

import hashlib
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urldefrag, unquote

from ebooklib import ITEM_DOCUMENT, epub
from lxml import html

from mcp_ebook_read.errors import AppError, ErrorCode
from mcp_ebook_read.schema.models import (
    ChunkRecord,
    ExtractedImage,
    Locator,
    OutlineNode,
    ParsedDocument,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_href_path(href: str) -> str:
    path, _ = urldefrag(href)
    normalized = unquote(path.strip().split("?", maxsplit=1)[0]).lstrip("/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _split_href(href: str | None) -> tuple[str, str]:
    if not href:
        return "", ""
    path, fragment = urldefrag(href)
    return _normalize_href_path(path), fragment.strip()


def _tag_name(node_tag: str) -> str:
    return node_tag.split("}", maxsplit=1)[-1].lower()


def _toc_title_href(entry: object) -> tuple[str, str]:
    if isinstance(entry, epub.Link):
        return _normalize_text(str(entry.title or "")), str(entry.href or "")
    if isinstance(entry, epub.Section):
        return _normalize_text(str(entry.title or "")), str(entry.href or "")

    title = _normalize_text(str(getattr(entry, "title", "") or ""))
    href = str(getattr(entry, "href", "") or "")
    return title, href


def _resolve_asset_href(base_href: str, raw_href: str) -> str:
    target, _ = urldefrag(raw_href.strip())
    if not target:
        return ""
    lowered = target.lower()
    if "://" in lowered or lowered.startswith(("data:", "mailto:")):
        return ""

    base_dir = posixpath.dirname(_normalize_href_path(base_href))
    if target.startswith("/"):
        joined = target.lstrip("/")
    else:
        joined = posixpath.normpath(posixpath.join(base_dir, target))
    return _normalize_href_path(joined)


def _parse_dimension(raw: object) -> int | None:
    if raw is None:
        return None
    matched = re.match(r"^\s*(\d+)", str(raw))
    if not matched:
        return None
    try:
        return int(matched.group(1))
    except ValueError:
        return None


def _guess_extension(href: str, media_type: str | None) -> str:
    suffix = Path(href).suffix.lower()
    if suffix:
        return suffix

    media_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
        "image/bmp": ".bmp",
    }
    return media_map.get((media_type or "").lower(), ".bin")


def _is_image_asset(href: str, media_type: str | None) -> bool:
    if media_type and media_type.lower().startswith("image/"):
        return True
    return Path(href).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".svg",
    }


def _extract_figcaption_text(img_node: html.HtmlElement) -> str:
    figure_nodes = img_node.xpath("ancestor::figure[1]")
    if not figure_nodes:
        return ""
    return _normalize_text(" ".join(figure_nodes[0].xpath(".//figcaption//text()")))


class EbooklibEpubParser:
    """Single path EPUB parser."""

    method = "ebooklib"

    def parse(self, epub_path: str, doc_id: str) -> ParsedDocument:
        path = Path(epub_path)
        if not path.exists():
            raise AppError(
                ErrorCode.INGEST_DOC_NOT_FOUND,
                f"Document not found: {epub_path}",
            )

        try:
            book = epub.read_epub(str(path))
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                ErrorCode.INGEST_EPUB_PARSE_FAILED,
                f"Failed to parse EPUB: {epub_path}",
            ) from exc

        title = path.stem
        meta_title = book.get_metadata("DC", "title")
        if meta_title and meta_title[0] and meta_title[0][0]:
            title = str(meta_title[0][0])

        spine_href_to_id: dict[str, str] = {}
        for spine_entry in book.spine:
            spine_id = str(spine_entry[0])
            item = book.get_item_with_id(spine_id)
            if item is None or item.get_type() != ITEM_DOCUMENT:
                continue
            item_name = _normalize_href_path(str(item.get_name() or spine_id))
            if item_name:
                spine_href_to_id[item_name] = spine_id

        outline: list[OutlineNode] = []
        toc_paths: dict[str, list[str]] = {}
        toc_counter = 0

        def register_toc_entry(
            node_title: str, node_href: str, level: int, path: list[str]
        ) -> OutlineNode:
            nonlocal toc_counter
            href_path, fragment = _split_href(node_href)
            spine_ref = spine_href_to_id.get(href_path, href_path or None)
            node_id = node_href or f"toc-{toc_counter}"
            toc_counter += 1
            if href_path:
                if href_path not in toc_paths:
                    toc_paths[href_path] = path
                if fragment:
                    toc_paths[f"{href_path}#{fragment}"] = path
            return OutlineNode(
                id=node_id,
                title=node_title or (href_path or f"section-{toc_counter}"),
                level=level,
                spine_ref=spine_ref,
            )

        def build_outline(
            entries: Sequence[object], level: int, parent_path: list[str]
        ) -> list[OutlineNode]:
            nodes: list[OutlineNode] = []
            for entry in entries:
                if isinstance(entry, tuple) and len(entry) == 2:
                    head, children = entry
                    head_title, head_href = _toc_title_href(head)
                    path = parent_path + [head_title] if head_title else parent_path
                    if head_title or head_href:
                        node = register_toc_entry(head_title, head_href, level, path)
                        node.children = build_outline(
                            list(children), level + 1, path.copy()
                        )
                        nodes.append(node)
                    else:
                        nodes.extend(build_outline(list(children), level, parent_path))
                    continue

                entry_title, entry_href = _toc_title_href(entry)
                if not entry_title and not entry_href:
                    continue
                path = parent_path + [entry_title] if entry_title else parent_path
                nodes.append(register_toc_entry(entry_title, entry_href, level, path))
            return nodes

        outline = build_outline(list(book.toc), level=1, parent_path=[])
        chunks: list[ChunkRecord] = []
        markdown_parts: list[str] = []
        extracted_images: list[ExtractedImage] = []

        asset_map: dict[str, tuple[bytes, str | None]] = {}
        get_items = getattr(book, "get_items", None)
        if callable(get_items):
            for asset in get_items():
                asset_name_getter = getattr(asset, "get_name", None)
                asset_content_getter = getattr(asset, "get_content", None)
                if not callable(asset_name_getter) or not callable(
                    asset_content_getter
                ):
                    continue
                normalized_name = _normalize_href_path(str(asset_name_getter() or ""))
                if not normalized_name:
                    continue
                try:
                    payload = bytes(asset_content_getter())
                except Exception:  # noqa: BLE001
                    continue
                media_type = str(getattr(asset, "media_type", "") or "") or None
                asset_map[normalized_name] = (payload, media_type)

        for spine_index, spine_entry in enumerate(book.spine):
            spine_id = str(spine_entry[0])
            item = book.get_item_with_id(spine_id)
            if item is None or item.get_type() != ITEM_DOCUMENT:
                continue

            try:
                dom = html.fromstring(item.get_content())
            except Exception as exc:  # noqa: BLE001
                raise AppError(
                    ErrorCode.INGEST_EPUB_PARSE_FAILED,
                    f"Invalid XHTML in spine item: {spine_id}",
                ) from exc

            spine_href = _normalize_href_path(str(item.get_name() or spine_id))
            spine_path = toc_paths.get(spine_href, [spine_id])
            if not spine_path:
                spine_path = [spine_id]

            def resolve_section_path(
                *,
                heading_stack: list[str],
                level: int,
                heading_text: str,
                anchor: str,
            ) -> list[str]:
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(heading_text)

                toc_key = (
                    f"{spine_href}#{anchor}" if spine_href and anchor else spine_href
                )
                resolved = toc_paths.get(toc_key)
                if resolved is None:
                    resolved = toc_paths.get(spine_href, [])
                    if heading_stack:
                        if resolved and heading_stack[0] == resolved[-1]:
                            resolved = resolved[:-1] + heading_stack
                        else:
                            resolved = resolved + heading_stack
                return resolved or heading_stack.copy() or spine_path

            body_nodes = dom.xpath("//body")
            scope = body_nodes[0] if body_nodes else dom
            content_nodes = scope.xpath(
                ".//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6 | "
                ".//p | .//li[not(descendant::p)] | "
                ".//blockquote[not(descendant::p)] | .//pre"
            )

            heading_path: list[str] = []
            current_section_path = spine_path.copy()
            current_anchor = ""
            current_paragraphs: list[str] = []
            section_counter = 0

            def flush_section() -> None:
                nonlocal section_counter
                if not current_paragraphs:
                    return

                raw_text = "\n\n".join(current_paragraphs)
                search_text = _normalize_text(raw_text)
                chunk_source = (
                    f"{doc_id}:{spine_index}:{section_counter}:{current_section_path}"
                    f":{current_anchor}"
                )
                chunk_id = hashlib.sha1(
                    chunk_source.encode(),
                    usedforsecurity=False,
                ).hexdigest()[:16]
                epub_locator = {"spine_id": spine_id}
                if spine_href:
                    epub_locator["href"] = spine_href
                if current_anchor:
                    epub_locator["anchor"] = current_anchor

                locator = Locator(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    section_path=current_section_path,
                    epub_locator=epub_locator,
                    method=self.method,
                    confidence=None,
                )
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        order_index=len(chunks),
                        section_path=current_section_path,
                        text=raw_text,
                        search_text=search_text,
                        locator=locator,
                        method=self.method,
                    )
                )
                markdown_parts.append(f"# {' / '.join(current_section_path)}")
                markdown_parts.append(raw_text)
                section_counter += 1

            for node in content_nodes:
                tag_name = _tag_name(str(node.tag))
                if re.fullmatch(r"h[1-6]", tag_name):
                    flush_section()

                    heading_text = _normalize_text(" ".join(node.itertext()))
                    if not heading_text:
                        current_anchor = _normalize_text(node.get("id") or "")
                        continue

                    level = int(tag_name[1])
                    anchor = _normalize_text(node.get("id") or "")
                    current_section_path = resolve_section_path(
                        heading_stack=heading_path,
                        level=level,
                        heading_text=heading_text,
                        anchor=anchor,
                    )
                    current_anchor = anchor
                    current_paragraphs = []
                    continue

                text = _normalize_text(" ".join(node.itertext()))
                if not text:
                    continue
                if not current_paragraphs or current_paragraphs[-1] != text:
                    current_paragraphs.append(text)

            flush_section()

            image_scan_nodes = scope.xpath(
                ".//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6 | .//img"
            )
            image_heading_path: list[str] = []
            image_section_path = spine_path.copy()
            image_anchor = ""

            for node in image_scan_nodes:
                tag_name = _tag_name(str(node.tag))
                if re.fullmatch(r"h[1-6]", tag_name):
                    heading_text = _normalize_text(" ".join(node.itertext()))
                    if not heading_text:
                        image_anchor = _normalize_text(node.get("id") or "")
                        continue

                    level = int(tag_name[1])
                    anchor = _normalize_text(node.get("id") or "")
                    image_section_path = resolve_section_path(
                        heading_stack=image_heading_path,
                        level=level,
                        heading_text=heading_text,
                        anchor=anchor,
                    )
                    image_anchor = anchor
                    continue

                src = str(node.get("src") or "").strip()
                if not src:
                    continue
                resolved_href = _resolve_asset_href(spine_href, src)
                if not resolved_href:
                    continue

                payload = asset_map.get(resolved_href)
                media_type = payload[1] if payload else None
                if not _is_image_asset(resolved_href, media_type):
                    continue
                if payload is None:
                    continue

                alt = _normalize_text(node.get("alt") or "")
                caption = _extract_figcaption_text(node)
                width = _parse_dimension(node.get("width"))
                height = _parse_dimension(node.get("height"))
                anchor = _normalize_text(node.get("id") or "") or image_anchor

                image_source = (
                    f"{doc_id}:{spine_index}:{len(extracted_images)}:"
                    f"{resolved_href}:{image_section_path}:{anchor}"
                )
                image_id = hashlib.sha1(
                    image_source.encode(),
                    usedforsecurity=False,
                ).hexdigest()[:16]
                extension = _guess_extension(resolved_href, media_type)
                extracted_images.append(
                    ExtractedImage(
                        image_id=image_id,
                        doc_id=doc_id,
                        order_index=len(extracted_images),
                        section_path=image_section_path.copy(),
                        spine_id=spine_id,
                        href=resolved_href,
                        anchor=anchor or None,
                        alt=alt or None,
                        caption=caption or None,
                        media_type=media_type,
                        extension=extension,
                        width=width,
                        height=height,
                        data=payload[0],
                        source=self.method,
                    )
                )

        return ParsedDocument(
            title=title,
            parser_chain=[self.method],
            metadata={
                "spine_items": len(book.spine),
                "toc_nodes": toc_counter,
                "chunking": "heading_sections",
                "images_extracted": len(extracted_images),
            },
            outline=outline,
            chunks=chunks,
            images=extracted_images,
            reading_markdown="\n\n".join(markdown_parts),
            raw_artifacts={},
            overall_confidence=None,
        )
