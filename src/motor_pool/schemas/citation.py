"""Citation and source-locator schemas.

Every retrieved chunk and every answer claim carries a Citation so the system
can point back to the exact spot in the manual. The anchor HMMWV TMs (the
-280 family) use classic chapter / section / paragraph numbering, not the
modern MIL-STD-40051 Work Package format, so the locator models a paragraph.

Forward-compat seam: `Locator` is kept as a one-member alias today. When a
Work-Package-format manual is added later, define a `WorkPackageLocator` with
its own `type` literal and switch `Locator` to a discriminated union:

    Locator = Annotated[
        Union[ParagraphLocator, WorkPackageLocator],
        Field(discriminator="type"),
    ]

Nothing else has to change because callers already read `citation.locator`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParagraphLocator(BaseModel):
    """A location expressed as chapter / section / paragraph, as printed.

    Paragraph numbers reset per chapter and include decimal change-page
    insertions (for example "2-104.1"). The raw string is the key. Never parse
    it to an integer; that loses information and breaks citation uniqueness.
    """

    type: Literal["paragraph"] = "paragraph"
    chapter: str = Field(description='Chapter as printed, e.g. "2". String, never math.')
    section: str | None = Field(
        default=None, description='Section roman numeral as printed, e.g. "IV".'
    )
    paragraph: str | None = Field(
        default=None, description='Raw paragraph key as printed, e.g. "2-104.1".'
    )


# One-member alias today; see module docstring for the discriminated-union seam.
Locator = ParagraphLocator


class Citation(BaseModel):
    """A pointer back to the exact location in a source TM that supports a claim."""

    source_doc_id: str = Field(description='Canonical TM id, e.g. "TM-9-2320-280-10".')
    source_doc_title: str
    distribution_statement: str = "Distribution Statement A"
    edition_date: str = Field(description="Pinned per the exact ingested PDF edition.")
    locator: Locator

    table_ref: str | None = Field(default=None, description='e.g. "Table 1-7".')
    figure_ref: str | None = Field(default=None, description='e.g. "Figure 8".')

    tm_page_label: str = Field(
        description='Printed page header, human-facing, e.g. "2-72".'
    )
    pdf_page_index: int = Field(
        description="0-based physical PDF page, for re-extraction and debugging."
    )
    source_pdf_sha256: str = Field(description="Provenance of the exact ingested file.")
    chunk_id: str = Field(description="Id of the chunk this citation was drawn from.")
