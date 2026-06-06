"""Citation canonicalization, shared by the data-gen validator and the eval scorer.

Comparing citations by raw string punishes harmless label differences such as
"Para 2-14" versus "Section 2-14". These functions reduce a citation to a
canonical (doc, locator, page) tuple so the comparison is on the identifier, not
the label. The same module is imported by eval, so train-time acceptance and
eval scoring can never drift apart.

A label is stripped only when it is followed by a real separator and more text,
so appendix-style identifiers (the HMMWV -10 TMs run appendices A through P with
keys like "P-1", and a bare appendix label "P") survive untouched.
"""

from __future__ import annotations

import re

# Leading locator label words. Longer alternatives come first so "section" wins
# over "sec"/"s" and "para"/"page" over "p". A label is only stripped when at
# least one separator follows it, so "P-1" and a bare "P" are never stripped.
_LABEL_RE = re.compile(
    r"^(?:paragraphs?|para|sections?|sect|sec|chapters?|chap|chp|pages?|pg|p)"
    r"[\s.:#)]+",
    re.IGNORECASE,
)


def normalize_token(raw: str) -> str:
    """Strip a leading locator label from an identifier, conservatively.

    >>> normalize_token("Para 2-14")
    '2-14'
    >>> normalize_token("Section 2-14")
    '2-14'
    >>> normalize_token("p. 2-72")
    '2-72'
    >>> normalize_token("2-104.1")
    '2-104.1'
    >>> normalize_token("P-1")        # appendix paragraph, label not stripped
    'p-1'
    >>> normalize_token("P")          # bare appendix label, preserved
    'p'
    """
    s = raw.strip().lower()
    s = _LABEL_RE.sub("", s)
    return re.sub(r"\s+", " ", s.strip(" .:#)"))


def normalize_page(raw: str) -> str:
    """Canonicalize a printed page label. No label-stripping (pages are not labelled).

    >>> normalize_page("2-72")
    '2-72'
    >>> normalize_page("P-3")
    'p-3'
    """
    return re.sub(r"\s+", " ", raw.strip().lower())


def normalize_doc_id(raw: str) -> str:
    """Canonicalize a TM id so spacing/hyphen variants compare equal.

    >>> normalize_doc_id("TM 9-2320-280-10") == normalize_doc_id("TM-9-2320-280-10")
    True
    """
    return re.sub(r"[\s_]+", "-", raw.strip().upper())


def canonicalize_citation(citation) -> tuple[str, str, str]:
    """Reduce a Citation to a canonical (doc_id, locator_id, page) tuple."""
    locator = citation.locator
    identifier = locator.paragraph or locator.section or locator.chapter or ""
    return (
        normalize_doc_id(citation.source_doc_id),
        normalize_token(identifier),
        normalize_page(citation.tm_page_label),
    )
