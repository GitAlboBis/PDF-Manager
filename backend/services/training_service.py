"""
training_service.py — Training data management for RAG confidence boosting.

Provides utilities for loading labeled training examples from the database
and using them to boost confidence scores during RAG field extraction.
backend/services/training_service.py — TrainingService for intelligent field completion.

The TrainingService uses labeled training examples (stored in the
``training_examples`` database table) to:

1. **Auto-generate missing emails** using the pattern
   ``firstname@domain`` where *domain* is detected from training email examples
   (e.g. ``john@example.com`` → domain ``example.com``).

2. **Fill blank fields** from matching training examples — the most common
   non-blank value seen for a given field is applied.

3. **Correct incorrect values** by comparing extracted values against the
   training consensus and replacing low-confidence mismatches.

4. **Handle edge cases**: multi-word names (use first word only), special
   characters (normalise/strip), partial emails (complete them).

Public API
----------
* ``extract_domain_pattern(training_emails)`` → ``str``
* ``generate_email(name, domain)`` → ``str``
* ``fill_blank_fields(extracted_fields, training_data)`` → ``list[dict]``
* ``apply_training(extracted_fields)`` → ``list[dict]``  (fetches DB data itself)
"""

from __future__ import annotations

import difflib
import logging
import logging
import re
import unicodedata
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String similarity
# ---------------------------------------------------------------------------

def _string_similarity(a: str, b: str) -> float:
    """Return a similarity ratio in [0.0, 1.0] between two strings.

    Uses ``difflib.SequenceMatcher`` with case-insensitive comparison.
    """
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()
# Fields we recognise and process
_ALL_FIELDS = [
    "Name",
    "Street Address",
    "City",
    "State",
    "Zip Code",
    "Home Phone",
    "Cell Phone",
    "Work Phone",
    "Email",
]

# Confidence level we assign when a value comes from training data
_TRAINING_CONFIDENCE = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Strip whitespace, NFC-normalise unicode, collapse inner whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_word(name: str) -> str:
    """Return the first word of *name*, lower-cased, letters only.

    Handles multi-word names and strips non-letter characters so that
    special characters from OCR noise don't pollute generated emails.

    Examples
    --------
    >>> _first_word("Rahul Misra")
    'rahul'
    >>> _first_word("  John  Doe  ")
    'john'
    >>> _first_word("María García")
    'mara'  # non-ASCII characters stripped; 'María' → 'Mara' → 'mara'
    """
    name = _normalise_text(name)
    first = name.split()[0] if name.split() else name
    # Keep ASCII letters only to produce a safe email local-part
    ascii_only = first.encode("ascii", errors="ignore").decode("ascii")
    letters_only = re.sub(r"[^a-zA-Z]", "", ascii_only)
    return letters_only.lower()


def _is_blank(value: str | None) -> bool:
    """Return True if *value* is None, empty, or whitespace-only."""
    return not (value or "").strip()


# ---------------------------------------------------------------------------
# TrainingService
# ---------------------------------------------------------------------------

class TrainingService:
    """Manages training examples and computes confidence boosts.

    This service is intentionally kept free of Flask imports so it can be
    unit-tested without an application context.  Callers are responsible for
    passing training examples retrieved from the database.
    """

    # Similarity threshold above which an extracted value is considered
    # consistent with a training example → confidence boost applied.
    HIGH_SIMILARITY_THRESHOLD = 0.80

    # Minimum confidence boost applied when an extracted value closely matches
    # a training example.
    CONFIDENCE_BOOST = 0.10

    # Confidence level assigned when an empty extracted value is replaced by
    # a training example value.
    FALLBACK_CONFIDENCE = 0.75

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def boost_confidence(
        self,
        field_name: str,
        extracted_value: str,
        training_examples: list[dict[str, Any]],
    ) -> tuple[str, float, bool]:
        """Apply training data to improve an extracted field value.

        Args:
            field_name:         The name of the field being extracted.
            extracted_value:    The value produced by the RAG pipeline.
            training_examples:  A list of dicts with ``field_name`` and
                                ``correct_value`` keys (as returned by
                                ``TrainingExample.to_dict()``).

        Returns:
            A 3-tuple ``(value, confidence_delta, used_training)`` where:

            * ``value`` — the (possibly updated) field value
            * ``confidence_delta`` — amount to add to the base confidence
              (may be negative if not used)
            * ``used_training`` — ``True`` if training data influenced the
              result
        """
        relevant = [
            ex["correct_value"]
            for ex in training_examples
            if ex["field_name"] == field_name and ex.get("correct_value")
        ]

        if not relevant:
            return extracted_value, 0.0, False

        # When the extracted value is blank, use the first training example
        # directly (no similarity comparison possible against empty string).
        if not (extracted_value or "").strip():
            return relevant[0], self.FALLBACK_CONFIDENCE, True

        best_match, best_score = self._best_match(extracted_value, relevant)

        if best_score >= self.HIGH_SIMILARITY_THRESHOLD:
            # Extracted value is consistent with training → boost confidence
            return extracted_value, self.CONFIDENCE_BOOST, True

        return extracted_value, 0.0, False

    def apply_training_to_results(
        self,
        rag_results: list[dict[str, Any]],
        training_examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply training examples to a list of RAG extraction results.

        Each result dict is expected to have ``field_name``, ``field_value``,
        and ``confidence`` keys (as produced by
        :meth:`RAGService.extract_fields`).

        Returns the same list with ``confidence`` values updated in-place.
        """
        if not training_examples:
            return rag_results

        for item in rag_results:
            value, delta, used = self.boost_confidence(
                item["field_name"],
                item.get("field_value", "") or "",
                training_examples,
            )
            if used:
                item["field_value"] = value
                # delta is always > 0 when used=True (CONFIDENCE_BOOST or
                # FALLBACK_CONFIDENCE), so we always enter this branch.
                if delta > 0:
                    item["confidence"] = round(
                        min(1.0, item["confidence"] + delta), 4
                    )
                item["training_boosted"] = True

        return rag_results

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _best_match(
        target: str, candidates: list[str]
    ) -> tuple[str | None, float]:
        """Return the candidate with the highest similarity to *target*."""
        best_candidate: str | None = None
        best_score = 0.0
        for candidate in candidates:
            score = _string_similarity(target, candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate, best_score
    """Intelligent field completion using training data patterns."""

    # ------------------------------------------------------------------
    # Domain pattern detection
    # ------------------------------------------------------------------

    def extract_domain_pattern(self, training_emails: list[str]) -> str:
        """Extract the most common email domain from *training_emails*.

        Returns the domain string without the ``@`` prefix (e.g. ``"example.com"``),
        or an empty string when no valid email is found.

        Examples
        --------
        >>> svc = TrainingService()
        >>> svc.extract_domain_pattern(["john@example.com", "jane@example.com"])
        'example.com'
        """
        domains: list[str] = []
        for email in training_emails:
            email = (email or "").strip()
            if "@" in email:
                domain = email.split("@", 1)[1].strip().lower()
                if domain and "." in domain:
                    domains.append(domain)

        if not domains:
            return ""

        # Most common domain wins; ties broken by alphabetical order for determinism
        counter = Counter(domains)
        most_common = counter.most_common()
        top_count = most_common[0][1]
        candidates = sorted(d for d, c in most_common if c == top_count)
        return candidates[0]

    # ------------------------------------------------------------------
    # Email generation
    # ------------------------------------------------------------------

    def generate_email(self, name: str, domain: str) -> str:
        """Generate an email address from the first name and *domain*.

        The local-part is derived from the first word of *name* with all
        non-ASCII and non-letter characters removed.

        Returns an empty string when either *name* or *domain* is empty,
        or when the resulting local-part would be empty.

        Examples
        --------
        >>> svc = TrainingService()
        >>> svc.generate_email("Rahul Misra", "example.com")
        'rahul@example.com'
        >>> svc.generate_email("", "example.com")
        ''
        """
        if not (name or "").strip() or not (domain or "").strip():
            return ""
        local = _first_word(name)
        if not local:
            return ""
        domain = domain.strip().lower().lstrip("@")
        return f"{local}@{domain}"

    # ------------------------------------------------------------------
    # Field completion
    # ------------------------------------------------------------------

    def fill_blank_fields(
        self,
        extracted_fields: list[dict[str, Any]],
        training_data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fill blank / incorrect fields using *training_data* patterns.

        Parameters
        ----------
        extracted_fields:
            List of ``{"field_name": str, "field_value": str,
            "confidence": float, ...}`` dicts as returned by RAGService.

        training_data:
            List of ``{"field_name": str, "field_value": str}`` dicts from
            the ``training_examples`` database table.

        Returns
        -------
        A new list of field dicts with missing/incorrect values filled in.
        Each corrected field gains a ``"confidence_source"`` key set to
        ``"training"`` so the caller knows where the value came from.
        """
        # Group training data by field_name → list of non-blank values
        training_by_field: dict[str, list[str]] = {}
        for row in training_data:
            fname = (row.get("field_name") or "").strip()
            fval = (row.get("field_value") or "").strip()
            if fname and fval:
                training_by_field.setdefault(fname, []).append(fval)

        # Collect training email values for domain detection
        training_emails = training_by_field.get("Email", [])
        domain = self.extract_domain_pattern(training_emails)

        # Index extracted fields for easy lookup
        fields_by_name: dict[str, dict[str, Any]] = {}
        for f in extracted_fields:
            fname = (f.get("field_name") or "").strip()
            if fname:
                fields_by_name[fname] = f

        result: list[dict[str, Any]] = []
        for f in extracted_fields:
            field = dict(f)  # copy so we don't mutate the original
            fname = (field.get("field_name") or "").strip()
            fval = (field.get("field_value") or "").strip()

            if fname == "Email":
                field = self._fill_email_field(
                    field, fval, domain, fields_by_name, training_by_field
                )
            elif _is_blank(fval) and fname in training_by_field:
                # Fill from most common training value
                common_val = self._most_common(training_by_field[fname])
                if common_val:
                    field["field_value"] = common_val
                    field["confidence"] = _TRAINING_CONFIDENCE
                    field["confidence_source"] = "training"
            elif not _is_blank(fval) and fname in training_by_field:
                # Overwrite low-similarity incorrect values with training data
                common_val = self._most_common(training_by_field[fname])
                if common_val and _string_similarity(fval, common_val) < 0.60:
                    logger.info(
                        "TrainingService: overwriting field '%s' value '%s' "
                        "→ '%s' (similarity %.2f < 0.60)",
                        fname, fval, common_val,
                        _string_similarity(fval, common_val),
                    )
                    field["field_value"] = common_val
                    field["confidence"] = _TRAINING_CONFIDENCE
                    field["confidence_source"] = "training_corrected"

            result.append(field)

        return result

    # ------------------------------------------------------------------
    # Public apply entry point (fetches training data from the DB)
    # ------------------------------------------------------------------

    def apply_training(
        self, extracted_fields: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch training examples from the database and apply them.

        This is the primary entry-point called by the RAG extraction route.
        It imports Flask-SQLAlchemy models at call-time so the service remains
        importable without a Flask application context (e.g. in unit tests that
        inject *training_data* directly into ``fill_blank_fields``).

        Returns the corrected field list; falls back to *extracted_fields*
        unchanged when the database is unavailable.
        """
        try:
            from models import TrainingExample  # type: ignore[import]
            rows = TrainingExample.query.all()
            training_data = [
                {"field_name": r.field_name, "field_value": r.field_value or ""}
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "TrainingService: could not load training examples (%s); "
                "skipping training intelligence.",
                exc,
            )
            return extracted_fields

        if not training_data:
            return extracted_fields

        return self.fill_blank_fields(extracted_fields, training_data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _most_common(values: list[str]) -> str:
        """Return the most common non-blank value in *values*."""
        non_blank = [v for v in values if v.strip()]
        if not non_blank:
            return ""
        counter = Counter(non_blank)
        return counter.most_common(1)[0][0]

    def _fill_email_field(
        self,
        field: dict[str, Any],
        current_email: str,
        domain: str,
        fields_by_name: dict[str, dict[str, Any]],
        training_by_field: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Apply email-specific completion logic.

        Priority:
        1. If *current_email* is already a complete, valid email → keep it.
        2. If training emails exist and we know the domain:
           a. If the current email is blank → generate ``firstname@domain``.
           b. If the current email lacks a domain → complete it.
        3. If the email is still blank after step 2 → fill from training list.
        """
        if not _is_blank(current_email) and "@" in current_email:
            # Already looks like a complete email
            return field

        if domain:
            # Determine name to use for generation
            name_field = fields_by_name.get("Name", {})
            name_val = (name_field.get("field_value") or "").strip()

            if _is_blank(current_email):
                # Case (a): blank — generate from name + domain
                generated = self.generate_email(name_val, domain)
                if generated:
                    field["field_value"] = generated
                    field["confidence"] = _TRAINING_CONFIDENCE
                    field["confidence_source"] = "training_generated"
                    return field
            else:
                # Case (b): partial email (no @) — append domain
                local = re.sub(r"[^a-zA-Z0-9._+-]", "", current_email.strip())
                if local:
                    field["field_value"] = f"{local}@{domain}"
                    field["confidence"] = _TRAINING_CONFIDENCE
                    field["confidence_source"] = "training_completed"
                    return field

        # Fallback: fill from most common training email
        if training_by_field.get("Email"):
            common_email = self._most_common(training_by_field["Email"])
            if common_email and _is_blank(current_email):
                field["field_value"] = common_email
                field["confidence"] = _TRAINING_CONFIDENCE
                field["confidence_source"] = "training"

        return field
