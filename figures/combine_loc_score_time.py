#!/usr/bin/env python3
"""Combine the current Evolve PySR and BasicSR figures on one vector PDF page."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation


ROOT = Path(__file__).resolve().parent


def main():
    sources = [ROOT / "709715_simplification_population_context/trajectory.pdf",
               ROOT / "150815_simplification_population_context/trajectory.pdf"]
    pages = [PdfReader(source).pages[0] for source in sources]
    widths = [float(page.mediabox.width) for page in pages]
    heights = [float(page.mediabox.height) for page in pages]
    writer = PdfWriter()
    combined = writer.add_blank_page(width=sum(widths), height=max(heights))
    x = 0
    for page, width, height in zip(pages, widths, heights):
        combined.merge_transformed_page(
            page, Transformation().translate(tx=x, ty=max(heights) - height))
        x += width
    writer.add_metadata({"/Title": "Evolve PySR (left) and BasicSR (right): LOC, score, and generation"})
    output = ROOT / "pysr_basicsr_loc_score_time.pdf"
    with output.open("wb") as handle:
        writer.write(handle)
    print(output)


if __name__ == "__main__":
    main()
