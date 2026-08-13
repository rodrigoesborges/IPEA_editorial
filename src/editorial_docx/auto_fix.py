"""Deterministic auto-fix for citation/reference issues in DOCX files.

Builds fix actions from analysis findings and applies them to the document XML,
enabling closed-loop validation: analyze -> fix -> re-analyze -> compare.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from .citations_eval import CitationAnalysisSnapshot, analyze_document_file
from .docx_utils import (
    NS,
    _new_paragraph_like,
    _paragraph_text,
    _parse_xml,
    _replace_paragraph_text,
    _serialize_xml,
)
from .review_patterns import _ascii_fold

ADD_REFERENCE = "add_reference"
REMOVE_REFERENCE = "remove_reference"
STANDARDIZE_REFERENCE = "standardize_reference"

_REFERENCE_HEADING_RE = re.compile(
    r"\b(referencias|referencia|references|bibliografia)\b", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}(?:[/-]\d{2,4})?[a-z]?\b", flags=re.IGNORECASE)
_URL_SPLIT_RE = re.compile(r"\b(?:Dispon[ií]vel em|Acesso em)\s*:", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FixAction:
    """Handles single deterministic fix action."""

    kind: str
    author_key: str
    year: str
    label: str
    reference_text: str
    corrected_text: str = ""


@dataclass(frozen=True, slots=True)
class FixResult:
    """Handles result of applying fixes to a document."""

    actions_applied: tuple[FixAction, ...]
    actions_skipped: tuple[FixAction, ...]
    output_path: Path
    pre_finding_count: int
    post_finding_count: int


def _build_minimal_reference_text(author_key: str, year: str) -> str:
    """Builds a minimal ABNT-ish reference entry from author_key and year."""
    display = author_key.upper().strip() or "AUTOR"
    return f"{display}. {year}. [Referencia a ser completada]."


def _replace_last_year(text: str, new_year: str) -> str:
    """Replaces the last year-like pattern before URL section."""
    parts = _URL_SPLIT_RE.split(text, maxsplit=1)
    head = parts[0]
    matches = list(_YEAR_RE.finditer(head))
    if matches:
        last = matches[-1]
        head = head[: last.start()] + new_year + head[last.end() :]
    return head + ("".join(parts[1:]) if len(parts) > 1 else "")


def build_fix_actions(snapshot: CitationAnalysisSnapshot) -> list[FixAction]:
    """Converts analysis findings to fix actions."""
    actions: list[FixAction] = []
    for finding in snapshot.findings:
        if finding.kind == "missing_citation":
            actions.append(
                FixAction(
                    kind=ADD_REFERENCE,
                    author_key=finding.author_key,
                    year=finding.year,
                    label=finding.label,
                    reference_text=_build_minimal_reference_text(
                        finding.author_key, finding.year
                    ),
                )
            )
        elif finding.kind == "uncited_reference":
            actions.append(
                FixAction(
                    kind=REMOVE_REFERENCE,
                    author_key=finding.author_key,
                    year=finding.year,
                    label=finding.label,
                    reference_text=finding.raw_text,
                )
            )
        elif finding.kind == "probable_match":
            corrected = _replace_last_year(finding.raw_text, finding.year)
            actions.append(
                FixAction(
                    kind=STANDARDIZE_REFERENCE,
                    author_key=finding.author_key,
                    year=finding.year,
                    label=finding.label,
                    reference_text=finding.raw_text,
                    corrected_text=corrected,
                )
            )
    return actions


def _find_references_section(
    paragraphs: list[etree._Element],
) -> tuple[int, int]:
    """Returns (heading_idx, last_entry_idx) for the references section."""
    heading_idx = -1
    for idx, para in enumerate(paragraphs):
        text = _ascii_fold(_paragraph_text(para)).casefold().strip()
        if _REFERENCE_HEADING_RE.search(text):
            heading_idx = idx
            break

    if heading_idx == -1:
        return (-1, -1)

    last_entry_idx = heading_idx
    for idx in range(heading_idx + 1, len(paragraphs)):
        text = _paragraph_text(paragraphs[idx]).strip()
        if not text:
            continue
        folded = _ascii_fold(text).casefold()
        if _REFERENCE_HEADING_RE.search(folded):
            break
        last_entry_idx = idx

    return (heading_idx, last_entry_idx)


def _find_paragraph_by_text(
    paragraphs: list[etree._Element],
    target: str,
) -> int | None:
    """Finds the paragraph index whose text matches target."""
    target_folded = " ".join(_ascii_fold(target).casefold().split())

    for idx, para in enumerate(paragraphs):
        para_folded = " ".join(
            _ascii_fold(_paragraph_text(para)).casefold().split()
        )
        if para_folded == target_folded:
            return idx

    for idx, para in enumerate(paragraphs):
        para_folded = _ascii_fold(_paragraph_text(para)).casefold()
        if target_folded and target_folded in para_folded:
            return idx

    return None


def apply_fixes(
    input_path: Path,
    output_path: Path,
    actions: list[FixAction],
) -> tuple[tuple[FixAction, ...], tuple[FixAction, ...]]:
    """Applies fix actions to a DOCX file, writing the result to output_path."""
    with zipfile.ZipFile(input_path, "r") as zin:
        parts = {name: zin.read(name) for name in zin.namelist()}

    document_root = _parse_xml(parts["word/document.xml"])

    applied: list[FixAction] = []
    skipped: list[FixAction] = []

    for action in actions:
        if action.kind != STANDARDIZE_REFERENCE:
            continue
        if not action.corrected_text or action.corrected_text == action.reference_text:
            skipped.append(action)
            continue
        paragraphs = document_root.findall(".//w:p", namespaces=NS)
        match_idx = _find_paragraph_by_text(paragraphs, action.reference_text)
        if match_idx is None:
            skipped.append(action)
            continue
        _replace_paragraph_text(paragraphs[match_idx], action.corrected_text)
        applied.append(action)

    for action in actions:
        if action.kind != REMOVE_REFERENCE:
            continue
        paragraphs = document_root.findall(".//w:p", namespaces=NS)
        match_idx = _find_paragraph_by_text(paragraphs, action.reference_text)
        if match_idx is None:
            skipped.append(action)
            continue
        target = paragraphs[match_idx]
        parent = target.getparent()
        if parent is None:
            skipped.append(action)
            continue
        parent.remove(target)
        applied.append(action)

    add_actions = [a for a in actions if a.kind == ADD_REFERENCE]
    if add_actions:
        paragraphs = document_root.findall(".//w:p", namespaces=NS)
        heading_idx, last_entry_idx = _find_references_section(paragraphs)
        if heading_idx == -1 or last_entry_idx < heading_idx:
            skipped.extend(add_actions)
        else:
            template = paragraphs[last_entry_idx]
            parent = template.getparent()
            if parent is None:
                skipped.extend(add_actions)
            else:
                insert_point = template
                for action in add_actions:
                    new_para = _new_paragraph_like(
                        template, action.reference_text
                    )
                    insert_pos = list(parent).index(insert_point) + 1
                    parent.insert(insert_pos, new_para)
                    insert_point = new_para
                    applied.append(action)

    parts["word/document.xml"] = _serialize_xml(document_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, raw in parts.items():
            zout.writestr(name, raw)

    return (tuple(applied), tuple(skipped))


def auto_fix_document(
    input_path: Path,
    output_path: Path | None = None,
) -> FixResult:
    """Analyzes, builds fix actions, applies them, and re-analyzes."""
    pre_snapshot = analyze_document_file(input_path)
    actions = build_fix_actions(pre_snapshot)

    resolved_output = output_path or input_path.with_name(
        input_path.stem + ".fixed.docx"
    )

    applied, skipped = apply_fixes(input_path, resolved_output, actions)

    post_snapshot = analyze_document_file(resolved_output)

    return FixResult(
        actions_applied=applied,
        actions_skipped=skipped,
        output_path=resolved_output,
        pre_finding_count=len(pre_snapshot.findings),
        post_finding_count=len(post_snapshot.findings),
    )
