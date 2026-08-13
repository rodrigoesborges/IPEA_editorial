"""Deterministic evaluation harness for citation/reference consistency.

This module implements stages 0, 1, 3, 5, and 6 of the test pipeline:

* Stage 0 -- deterministic extraction (reuses ABNT parsers/matcher).
* Stage 1 -- gold derivation via set-diff of pre-edit vs post-edit snapshots.
* Stage 3 -- classification of LLM comments against the gold diff.
* Stage 5 -- JSON diff output + gold dataset artifact.
* Stage 6 -- deterministic auto-fix evaluation against human edits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .abnt_citation_parser import CitationCandidate, extract_citation_candidates
from .abnt_matcher import compare_citations_to_references
from .abnt_reference_parser import ParsedReferenceEntry, parse_reference_entry
from .agents.heuristics.references import NON_AUTHOR_REFERENCE_TOKENS
from .document_loader import LoadedDocument, load_document
from .models import (
    ReferenceAnchor,
    ReferenceBodyCitation,
    ReferenceEntryRecord,
    ReferencePipelineArtifact,
)
from .review_patterns import _ascii_fold, _is_non_body_reference_context, _ref_block_type

PROBABLE_KIND = "probable_match"

CITATION_AGENT_LABELS = frozenset({"referencias", "ref"})
CITATION_CATEGORY_LABELS = frozenset({"citation_match", "inconsistency", "citation_format"})


@dataclass(frozen=True, slots=True)
class CitationFinding:
    """Handles single deterministic finding."""

    kind: str
    author_key: str
    year: str
    label: str
    excerpt: str = ""
    raw_text: str = ""


@dataclass(frozen=True, slots=True)
class CitationAnalysisSnapshot:
    """Handles deterministic analysis of one document version."""

    citations: tuple[CitationCandidate, ...]
    references: tuple[ParsedReferenceEntry, ...]
    findings: tuple[CitationFinding, ...]


@dataclass(frozen=True, slots=True)
class CitationGoldDiff:
    """Handles diff between pre-edit and post-edit snapshots."""

    gold_positives: tuple[CitationFinding, ...]
    gold_negatives: tuple[CitationFinding, ...]
    new_in_post: tuple[CitationFinding, ...]


@dataclass(frozen=True, slots=True)
class ClassifiedComment:
    """Handles one LLM comment classified against gold."""

    comment: dict
    identity: tuple[str, str, str] | None
    classification: str


@dataclass(frozen=True, slots=True)
class CitationClassification:
    """Handles full classification result."""

    true_positives: tuple[ClassifiedComment, ...]
    fp_persisted: tuple[ClassifiedComment, ...]
    fp_novel: tuple[ClassifiedComment, ...]
    false_negatives: tuple[CitationFinding, ...]
    llm_only: tuple[ClassifiedComment, ...]
    deterministic_only: tuple[CitationFinding, ...]
    unmatched: tuple[ClassifiedComment, ...]


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    """Handles citation-level evaluation metrics."""

    tp: int
    fp_persisted: int
    fp_novel: int
    fn: int
    precision: float
    recall: float
    f1: float
    recall_missing: float
    recall_uncited: float
    recall_probable: float
    fp_structural: int


@dataclass(frozen=True, slots=True)
class AutoFixEvaluation:
    """Handles evaluation of deterministic auto-fix against human edits."""

    actions_applied: tuple[dict[str, object], ...]
    actions_skipped: tuple[dict[str, object], ...]
    fixed_output_path: str
    pre_finding_count: int
    fixed_finding_count: int
    agreed_with_human: tuple[CitationFinding, ...]
    over_fixed: tuple[CitationFinding, ...]
    missed_by_autofix: tuple[CitationFinding, ...]
    agreement_rate: float


@dataclass(frozen=True, slots=True)
class CitationEvalResult:
    """Handles complete evaluation result."""

    pre_snapshot: CitationAnalysisSnapshot
    post_snapshot: CitationAnalysisSnapshot
    diff: CitationGoldDiff
    classification: CitationClassification
    metrics: CitationMetrics
    auto_fix: AutoFixEvaluation | None = None


def _safe_div(numerator: float, denominator: float) -> float:
    """Handles safe division."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Handles f1."""
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _non_body_noop(*_args: object, **_kwargs: object) -> bool:
    """Handles noop non-body context check."""
    return False


def _identity_from_finding(finding: CitationFinding) -> tuple[str, str, str]:
    """Handles identity from finding."""
    return (finding.author_key, finding.year, finding.kind)


def analyze_loaded_document(doc: LoadedDocument) -> CitationAnalysisSnapshot:
    """Analyzes a loaded document for citation/reference consistency."""
    chunks = doc.chunks
    refs = doc.refs

    reference_heading_idx = -1
    for idx, ref_label in enumerate(refs):
        if _ref_block_type(ref_label) == "reference_heading":
            reference_heading_idx = idx
            break
    if reference_heading_idx == -1:
        for idx, chunk in enumerate(chunks):
            if _ascii_fold(chunk or "").casefold().strip() in {"referencias", "referencia"}:
                reference_heading_idx = idx
                break
    if reference_heading_idx == -1:
        reference_heading_idx = len(chunks)

    body_limit = reference_heading_idx

    citations = extract_citation_candidates(
        chunks,
        refs,
        body_limit,
        is_non_body_context=_is_non_body_reference_context,
        blocked_author_tokens=NON_AUTHOR_REFERENCE_TOKENS,
    )

    references: list[ParsedReferenceEntry] = []
    for idx in range(reference_heading_idx + 1, len(chunks)):
        if idx >= len(refs) or _ref_block_type(refs[idx]) != "reference_entry":
            continue
        parsed = parse_reference_entry(
            chunks[idx],
            blocked_author_tokens=NON_AUTHOR_REFERENCE_TOKENS,
        )
        if parsed is not None:
            references.append(parsed)

    match_result = compare_citations_to_references(list(citations), references)

    findings: list[CitationFinding] = []
    for citation in match_result.missing_citations:
        findings.append(
            CitationFinding(
                kind="missing_citation",
                author_key=citation.key[0],
                year=citation.key[1],
                label=citation.label,
                excerpt=citation.excerpt,
                raw_text=citation.excerpt,
            )
        )
    for entry in match_result.uncited_references:
        findings.append(
            CitationFinding(
                kind="uncited_reference",
                author_key=entry.author_key,
                year=entry.publication_year,
                label=entry.label,
                excerpt=entry.raw_text,
                raw_text=entry.raw_text,
            )
        )
    for match in match_result.probable_matches:
        findings.append(
            CitationFinding(
                kind=PROBABLE_KIND,
                author_key=match.citation.key[0],
                year=match.citation.key[1],
                label=match.citation.label,
                excerpt=match.citation.excerpt,
                raw_text=match.reference.raw_text,
            )
        )

    return CitationAnalysisSnapshot(
        citations=tuple(citations),
        references=tuple(references),
        findings=tuple(findings),
    )


def analyze_document_file(path: Path) -> CitationAnalysisSnapshot:
    """Analyzes a document file for citation/reference consistency."""
    doc = load_document(path)
    return analyze_loaded_document(doc)


def diff_snapshots(
    pre: CitationAnalysisSnapshot,
    post: CitationAnalysisSnapshot,
) -> CitationGoldDiff:
    """Diffs pre-edit and post-edit snapshots to derive gold labels."""
    pre_map: dict[tuple[str, str, str], CitationFinding] = {}
    for finding in pre.findings:
        pre_map[_identity_from_finding(finding)] = finding

    post_map: dict[tuple[str, str, str], CitationFinding] = {}
    for finding in post.findings:
        post_map[_identity_from_finding(finding)] = finding

    gold_positives = tuple(
        pre_map[key] for key in pre_map if key not in post_map
    )
    gold_negatives = tuple(
        pre_map[key] for key in pre_map if key in post_map
    )
    new_in_post = tuple(
        post_map[key] for key in post_map if key not in pre_map
    )

    return CitationGoldDiff(
        gold_positives=gold_positives,
        gold_negatives=gold_negatives,
        new_in_post=new_in_post,
    )


def canonical_citation_key_from_text(text: str) -> tuple[str, str] | None:
    """Extracts canonical (author_key, year) from free text."""
    if not text or not text.strip():
        return None

    candidates = extract_citation_candidates(
        [text],
        ["tipo=paragraph"],
        1,
        is_non_body_context=_non_body_noop,
        blocked_author_tokens=NON_AUTHOR_REFERENCE_TOKENS,
    )
    if candidates:
        return candidates[0].key

    parsed = parse_reference_entry(
        text,
        blocked_author_tokens=NON_AUTHOR_REFERENCE_TOKENS,
    )
    if parsed is not None:
        return parsed.key

    return None


def _guess_finding_kind(comment: dict) -> str:
    """Handles guess finding kind from a comment dict."""
    category = _ascii_fold(str(comment.get("category") or "")).casefold().strip()
    message = _ascii_fold(str(comment.get("message") or "")).casefold()
    suggested = _ascii_fold(str(comment.get("suggested_fix") or "")).casefold()
    if "incluir" in suggested:
        return "missing_citation"
    if "verificar estas obras" in suggested:
        return "uncited_reference"
    if "nao tem correspondencia" in message or "sem correspondencia clara" in message:
        return "missing_citation"
    if "nao foram localizadas" in message:
        return "uncited_reference"
    if "citation_format" in category:
        return "missing_citation"
    return PROBABLE_KIND


def extract_citation_comments_from_report(report: list[dict]) -> list[dict]:
    """Filters a relatorio.json to citation-relevant comments only."""
    picked: list[dict] = []
    for item in report:
        agent = _ascii_fold(str(item.get("agent") or "")).casefold().strip()
        category = _ascii_fold(str(item.get("category") or "")).casefold().strip()
        if agent not in CITATION_AGENT_LABELS:
            continue
        if category not in CITATION_CATEGORY_LABELS:
            continue
        picked.append(item)
    return picked


def classify_llm_comments(
    llm_comments: list[dict],
    diff: CitationGoldDiff,
    deterministic_snapshot: CitationAnalysisSnapshot,
) -> CitationClassification:
    """Classifies LLM comments against gold diff and deterministic findings."""
    gold_keys: set[tuple[str, str, str]] = {
        _identity_from_finding(f) for f in diff.gold_positives
    }
    negative_keys: set[tuple[str, str, str]] = {
        _identity_from_finding(f) for f in diff.gold_negatives
    }
    deterministic_keys: set[tuple[str, str, str]] = {
        _identity_from_finding(f) for f in deterministic_snapshot.findings
    }

    true_positives: list[ClassifiedComment] = []
    fp_persisted: list[ClassifiedComment] = []
    fp_novel: list[ClassifiedComment] = []
    llm_only: list[ClassifiedComment] = []
    unmatched: list[ClassifiedComment] = []
    matched_gold_ids: set[tuple[str, str, str]] = set()

    for comment in llm_comments:
        text_blob = " ".join(
            [
                str(comment.get("issue_excerpt") or ""),
                str(comment.get("suggested_fix") or ""),
                str(comment.get("message") or ""),
            ]
        )
        key = canonical_citation_key_from_text(text_blob)
        if key is None:
            unmatched.append(
                ClassifiedComment(
                    comment=comment,
                    identity=None,
                    classification="unmatched",
                )
            )
            continue

        kind = _guess_finding_kind(comment)
        identity = (key[0], key[1], kind)

        if identity in gold_keys:
            true_positives.append(
                ClassifiedComment(
                    comment=comment,
                    identity=identity,
                    classification="TP",
                )
            )
            matched_gold_ids.add(identity)
        elif identity in negative_keys:
            fp_persisted.append(
                ClassifiedComment(
                    comment=comment,
                    identity=identity,
                    classification="FP_persisted",
                )
            )
        elif identity in deterministic_keys:
            llm_only.append(
                ClassifiedComment(
                    comment=comment,
                    identity=identity,
                    classification="llm_only",
                )
            )
        else:
            fp_novel.append(
                ClassifiedComment(
                    comment=comment,
                    identity=identity,
                    classification="FP_novel",
                )
            )

    false_negatives = tuple(
        f for f in diff.gold_positives
        if _identity_from_finding(f) not in matched_gold_ids
    )

    matched_deterministic = gold_keys | negative_keys | {
        item.identity for item in llm_only if item.identity is not None
    }
    deterministic_only = tuple(
        f for f in deterministic_snapshot.findings
        if _identity_from_finding(f) not in matched_deterministic
    )

    return CitationClassification(
        true_positives=tuple(true_positives),
        fp_persisted=tuple(fp_persisted),
        fp_novel=tuple(fp_novel),
        false_negatives=false_negatives,
        llm_only=tuple(llm_only),
        deterministic_only=deterministic_only,
        unmatched=tuple(unmatched),
    )


def compute_citation_metrics(
    classification: CitationClassification,
    diff: CitationGoldDiff,
) -> CitationMetrics:
    """Computes precision/recall/F1 and per-kind recall from classification."""
    tp = len(classification.true_positives)
    fp_persisted = len(classification.fp_persisted)
    fp_novel = len(classification.fp_novel)
    fn = len(classification.false_negatives)

    precision = _safe_div(tp, tp + fp_persisted + fp_novel)
    recall = _safe_div(tp, tp + fn)
    f1 = _f1(precision, recall)

    gold_missing = sum(1 for f in diff.gold_positives if f.kind == "missing_citation")
    gold_uncited = sum(1 for f in diff.gold_positives if f.kind == "uncited_reference")
    gold_probable = sum(1 for f in diff.gold_positives if f.kind == PROBABLE_KIND)

    detected_missing = sum(
        1
        for item in classification.true_positives
        if item.identity is not None and item.identity[2] == "missing_citation"
    )
    detected_uncited = sum(
        1
        for item in classification.true_positives
        if item.identity is not None and item.identity[2] == "uncited_reference"
    )
    detected_probable = sum(
        1
        for item in classification.true_positives
        if item.identity is not None and item.identity[2] == PROBABLE_KIND
    )

    recall_missing = _safe_div(detected_missing, gold_missing)
    recall_uncited = _safe_div(detected_uncited, gold_uncited)
    recall_probable = _safe_div(detected_probable, gold_probable)

    fp_structural = sum(
        1
        for item in classification.fp_persisted
        if item.identity is not None and item.identity[2] == PROBABLE_KIND
    )

    return CitationMetrics(
        tp=tp,
        fp_persisted=fp_persisted,
        fp_novel=fp_novel,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        recall_missing=recall_missing,
        recall_uncited=recall_uncited,
        recall_probable=recall_probable,
        fp_structural=fp_structural,
    )


def evaluate_auto_fix(
    original_path: Path,
    post_snapshot: CitationAnalysisSnapshot,
    pre_snapshot: CitationAnalysisSnapshot,
) -> AutoFixEvaluation:
    """Runs deterministic auto-fix and compares against human edits."""
    from .auto_fix import auto_fix_document

    fix_result = auto_fix_document(original_path)
    fixed_snapshot = analyze_document_file(fix_result.output_path)

    pre_map: dict[tuple[str, str, str], CitationFinding] = {
        _identity_from_finding(f): f for f in pre_snapshot.findings
    }
    post_keys: set[tuple[str, str, str]] = {
        _identity_from_finding(f) for f in post_snapshot.findings
    }
    fixed_keys: set[tuple[str, str, str]] = {
        _identity_from_finding(f) for f in fixed_snapshot.findings
    }

    human_resolved = {k for k in pre_map if k not in post_keys}
    autofix_resolved = {k for k in pre_map if k not in fixed_keys}

    agreed = tuple(pre_map[k] for k in (human_resolved & autofix_resolved))
    over_fixed = tuple(pre_map[k] for k in (autofix_resolved - human_resolved))
    missed = tuple(pre_map[k] for k in (human_resolved - autofix_resolved))

    agreement = _safe_div(len(agreed), len(human_resolved))

    def _action_dict(action: object) -> dict[str, object]:
        return {
            "kind": getattr(action, "kind", ""),
            "author_key": getattr(action, "author_key", ""),
            "year": getattr(action, "year", ""),
            "label": getattr(action, "label", ""),
        }

    return AutoFixEvaluation(
        actions_applied=tuple(_action_dict(a) for a in fix_result.actions_applied),
        actions_skipped=tuple(_action_dict(a) for a in fix_result.actions_skipped),
        fixed_output_path=str(fix_result.output_path),
        pre_finding_count=fix_result.pre_finding_count,
        fixed_finding_count=fix_result.post_finding_count,
        agreed_with_human=agreed,
        over_fixed=over_fixed,
        missed_by_autofix=missed,
        agreement_rate=agreement,
    )


def build_diff_json(
    diff: CitationGoldDiff,
    classification: CitationClassification,
    metrics: CitationMetrics,
    *,
    model_name: str = "",
    run_label: str = "",
    source_document: str = "",
    auto_fix: AutoFixEvaluation | None = None,
) -> dict[str, object]:
    """Builds the stage-5 diff JSON payload."""
    def _comment_summary(item: ClassifiedComment) -> dict[str, object]:
        return {
            "agent": item.comment.get("agent") or "",
            "category": item.comment.get("category") or "",
            "message": item.comment.get("message") or "",
            "issue_excerpt": item.comment.get("issue_excerpt") or "",
            "suggested_fix": item.comment.get("suggested_fix") or "",
            "identity": list(item.identity) if item.identity else None,
            "classification": item.classification,
        }

    def _finding_summary(finding: CitationFinding) -> dict[str, object]:
        return {
            "kind": finding.kind,
            "author_key": finding.author_key,
            "year": finding.year,
            "label": finding.label,
            "excerpt": finding.excerpt[:300],
        }

    result: dict[str, object] = {
        "stage": "citations_diff_v1",
        "document": {
            "source_document": source_document,
            "model_name": model_name,
            "run_label": run_label,
        },
        "counts": {
            "gold_positives": len(diff.gold_positives),
            "gold_negatives": len(diff.gold_negatives),
            "new_in_post": len(diff.new_in_post),
            "tp": metrics.tp,
            "fp_persisted": metrics.fp_persisted,
            "fp_novel": metrics.fp_novel,
            "fn": metrics.fn,
            "llm_only": len(classification.llm_only),
            "deterministic_only": len(classification.deterministic_only),
            "unmatched": len(classification.unmatched),
        },
        "metrics": {
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "recall_missing": round(metrics.recall_missing, 4),
            "recall_uncited": round(metrics.recall_uncited, 4),
            "recall_probable": round(metrics.recall_probable, 4),
            "fp_structural": metrics.fp_structural,
        },
        "blocks": {
            "deterministic_only": [
                _finding_summary(f) for f in classification.deterministic_only
            ],
            "llm_only": [_comment_summary(c) for c in classification.llm_only],
            "agreement": [_comment_summary(c) for c in classification.true_positives],
            "false_positives_persisted": [
                _comment_summary(c) for c in classification.fp_persisted
            ],
            "false_positives_novel": [
                _comment_summary(c) for c in classification.fp_novel
            ],
            "false_negatives": [
                _finding_summary(f) for f in classification.false_negatives
            ],
            "unmatched_comments": [
                _comment_summary(c) for c in classification.unmatched
            ],
        },
    }

    if auto_fix is not None:
        result["auto_fix"] = {
            "actions_applied": list(auto_fix.actions_applied),
            "actions_skipped": list(auto_fix.actions_skipped),
            "fixed_output_path": auto_fix.fixed_output_path,
            "pre_finding_count": auto_fix.pre_finding_count,
            "fixed_finding_count": auto_fix.fixed_finding_count,
            "agreed_with_human": [
                _finding_summary(f) for f in auto_fix.agreed_with_human
            ],
            "over_fixed": [
                _finding_summary(f) for f in auto_fix.over_fixed
            ],
            "missed_by_autofix": [
                _finding_summary(f) for f in auto_fix.missed_by_autofix
            ],
            "agreement_rate": round(auto_fix.agreement_rate, 4),
        }

    return result


def build_gold_dataset(
    diff: CitationGoldDiff,
    classification: CitationClassification,
    *,
    source_document: str = "",
    report_path: str = "",
    model_name: str = "",
    run_label: str = "",
) -> dict[str, object]:
    """Builds a gold-dataset JSON compatible with gold_metrics.compute_gold_metrics."""
    annotations: list[dict[str, object]] = []

    for ordinal, item in enumerate(
        [*classification.true_positives, *classification.llm_only], start=1
    ):
        annotations.append(
            {
                "id": f"ref_{ordinal:04d}",
                "agent": item.comment.get("agent") or "",
                "category": item.comment.get("category") or "",
                "paragraph_index": item.comment.get("paragraph_index"),
                "issue_excerpt": item.comment.get("issue_excerpt") or "",
                "suggested_fix": item.comment.get("suggested_fix") or "",
                "model_comment": item.comment.get("message") or "",
                "label": "correto" if item.classification == "TP" else "parcial",
                "severity": "",
                "reviewer_note": "",
                "source": {
                    "document": source_document,
                    "report_path": report_path,
                    "model_name": model_name,
                    "run_label": run_label,
                },
            }
        )

    for ordinal, item in enumerate(
        [*classification.fp_persisted, *classification.fp_novel], start=1
    ):
        annotations.append(
            {
                "id": f"ref_fp_{ordinal:04d}",
                "agent": item.comment.get("agent") or "",
                "category": item.comment.get("category") or "",
                "paragraph_index": item.comment.get("paragraph_index"),
                "issue_excerpt": item.comment.get("issue_excerpt") or "",
                "suggested_fix": item.comment.get("suggested_fix") or "",
                "model_comment": item.comment.get("message") or "",
                "label": "incorreto",
                "severity": "",
                "reviewer_note": "",
                "source": {
                    "document": source_document,
                    "report_path": report_path,
                    "model_name": model_name,
                    "run_label": run_label,
                },
            }
        )

    missed_issues: list[dict[str, object]] = []
    for ordinal, finding in enumerate(classification.false_negatives, start=1):
        missed_issues.append(
            {
                "id": f"faltou_{ordinal:04d}",
                "agent": "referencias",
                "paragraph_index": None,
                "issue_excerpt": finding.label,
                "expected_fix": finding.raw_text[:500],
                "label": "faltou",
                "severity": "",
                "reviewer_note": "",
            }
        )

    return {
        "dataset_version": "1.0-citations",
        "label_taxonomy": {
            "comment_labels": ["correto", "parcial", "incorreto"],
            "severity_labels": ["alta", "media", "baixa"],
            "missed_issue_labels": ["faltou"],
        },
        "document": {
            "source_document": source_document,
            "report_path": report_path,
            "model_name": model_name,
            "run_label": run_label,
        },
        "summary": {
            "total_model_comments": len(annotations),
            "tp": len(classification.true_positives),
            "fp_persisted": len(classification.fp_persisted),
            "fp_novel": len(classification.fp_novel),
            "fn": len(classification.false_negatives),
            "gold_positives_count": len(diff.gold_positives),
            "gold_negatives_count": len(diff.gold_negatives),
        },
        "annotations": annotations,
        "missed_issues": missed_issues,
    }


def to_pipeline_artifact(snapshot: CitationAnalysisSnapshot) -> ReferencePipelineArtifact:
    """Converts a CitationAnalysisSnapshot to a ReferencePipelineArtifact."""
    body_citations: list[ReferenceBodyCitation] = [
        ReferenceBodyCitation(
            paragraph_index=c.paragraph_index,
            excerpt=c.excerpt,
            label=c.label,
            key=c.key,
        )
        for c in snapshot.citations
    ]

    reference_entries: list[ReferenceEntryRecord] = [
        ReferenceEntryRecord(
            paragraph_index=-1,
            raw_text=ref.raw_text,
            label=ref.label,
            key=ref.key,
            document_type=ref.document_type,
            publication_year=ref.publication_year,
        )
        for ref in snapshot.references
    ]

    missing_citations: list[ReferenceBodyCitation] = []
    uncited_references: list[ReferenceEntryRecord] = []
    exact_anchors: list[ReferenceAnchor] = []
    probable_anchors: list[ReferenceAnchor] = []

    for finding in snapshot.findings:
        if finding.kind == "missing_citation":
            missing_citations.append(
                ReferenceBodyCitation(
                    paragraph_index=-1,
                    excerpt=finding.excerpt,
                    label=finding.label,
                    key=(finding.author_key, finding.year),
                )
            )
        elif finding.kind == "uncited_reference":
            uncited_references.append(
                ReferenceEntryRecord(
                    paragraph_index=-1,
                    raw_text=finding.raw_text,
                    label=finding.label,
                    key=(finding.author_key, finding.year),
                )
            )
        elif finding.kind == PROBABLE_KIND:
            probable_anchors.append(
                ReferenceAnchor(
                    citation_paragraph_index=-1,
                    citation_excerpt=finding.excerpt,
                    citation_label=finding.label,
                    reference_label=finding.label,
                    status="probable",
                    confidence=0.8,
                )
            )

    return ReferencePipelineArtifact(
        body_citations=body_citations,
        reference_entries=reference_entries,
        exact_anchors=exact_anchors,
        probable_anchors=probable_anchors,
        missing_citations=missing_citations,
        uncited_references=uncited_references,
    )


def run_citation_eval(
    original_path: Path,
    final_path: Path,
    report_path: Path | None = None,
    *,
    model_name: str = "",
    run_label: str = "",
    run_auto_fix: bool = False,
) -> CitationEvalResult:
    """Runs the full citation evaluation pipeline."""
    pre_snapshot = analyze_document_file(original_path)
    post_snapshot = analyze_document_file(final_path)
    diff = diff_snapshots(pre_snapshot, post_snapshot)

    llm_comments: list[dict] = []
    if report_path is not None:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(report_data, list):
            llm_comments = extract_citation_comments_from_report(report_data)

    classification = classify_llm_comments(llm_comments, diff, pre_snapshot)
    metrics = compute_citation_metrics(classification, diff)

    auto_fix_eval: AutoFixEvaluation | None = None
    if run_auto_fix:
        auto_fix_eval = evaluate_auto_fix(
            original_path, post_snapshot, pre_snapshot
        )

    return CitationEvalResult(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        diff=diff,
        classification=classification,
        metrics=metrics,
        auto_fix=auto_fix_eval,
    )


def main() -> int:
    """Runs the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Avalia determinisiticamente a consistência citação/referência."
    )
    parser.add_argument("--original", type=Path, required=True, help="Documento pre-edit.")
    parser.add_argument("--final", type=Path, required=True, help="Documento post-edit.")
    parser.add_argument("--report", type=Path, default=None, help="relatorio.json do LLM.")
    parser.add_argument("--output", type=Path, default=None, help="JSON de saida do diff.")
    parser.add_argument("--gold-output", type=Path, default=None, help="JSON do dataset ouro.")
    parser.add_argument("--model-name", default="", help="Nome do modelo.")
    parser.add_argument("--run-label", default="", help="Rotulo da rodada.")
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        default=False,
        help="Executa auto-fix deterministico e avalia contra a edicao humana.",
    )
    args = parser.parse_args()

    result = run_citation_eval(
        args.original,
        args.final,
        args.report,
        model_name=args.model_name,
        run_label=args.run_label,
        run_auto_fix=args.auto_fix,
    )

    source_doc = str(args.original)

    diff_payload = build_diff_json(
        result.diff,
        result.classification,
        result.metrics,
        model_name=args.model_name,
        run_label=args.run_label,
        source_document=source_doc,
        auto_fix=result.auto_fix,
    )

    gold_payload = build_gold_dataset(
        result.diff,
        result.classification,
        source_document=source_doc,
        report_path=str(args.report) if args.report else "",
        model_name=args.model_name,
        run_label=args.run_label,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(diff_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(args.output)

    if args.gold_output:
        args.gold_output.parent.mkdir(parents=True, exist_ok=True)
        args.gold_output.write_text(
            json.dumps(gold_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(args.gold_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
