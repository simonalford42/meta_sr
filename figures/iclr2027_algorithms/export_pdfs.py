"""Check the compiled proof sheet and export cropped vector algorithm PDFs."""
from pathlib import Path
import json
import pymupdf

ROOT = Path(__file__).resolve().parent
NAMES = (
    "meta_evolution", "pysr", "basicsr", "evolved_mutation",
    "evolved_survival", "evolved_selection", "evolved_loss",
)


def main():
    for name in ("algorithms", "meta_compact"):
        log = (ROOT / f"{name}.log").read_text()
        for diagnostic in ("Overfull", "Missing character"):
            assert diagnostic not in log, (name, diagnostic)
    source = pymupdf.open(ROOT / "algorithms.pdf")
    assert len(source) == len(NAMES), f"Expected 7 pages, got {len(source)}"
    previews = ROOT / "previews"
    previews.mkdir(exist_ok=True)
    report = []
    compact = pymupdf.open(ROOT / "meta_compact.pdf")
    assert len(compact) == 1
    inputs = [(source, i, name) for i, name in enumerate(NAMES)]
    inputs.append((compact, 0, "meta_evolution_compact"))
    for document, i, name in inputs:
        page = document[i]
        text = page.get_text()
        assert f"Algorithm {i+1}" in text, (name, text[:100])
        assert "return" in text.lower(), f"Missing return in {name}"
        blocks = page.get_text("blocks")
        box = pymupdf.Rect(blocks[0][:4])
        for block in blocks[1:]:
            box |= pymupdf.Rect(block[:4])
        for drawing in page.get_drawings():
            box |= drawing["rect"]
        assert box.x0 >= 100 and box.x1 <= 512, (name, "horizontal overflow", box)
        assert box.y1 < 750, (name, "vertical overflow", box)
        box += (-5, -5, 5, 5)
        target = pymupdf.open()
        target.new_page(width=box.width, height=box.height).show_pdf_page(
            pymupdf.Rect(0, 0, box.width, box.height), document, i, clip=box
        )
        target.save(ROOT / f"{name}.pdf", garbage=4, deflate=True)
        target[0].get_pixmap(matrix=pymupdf.Matrix(1.8, 1.8)).save(previews / f"{name}.png")
        report.append({"algorithm": name, "page": i+1,
                       "width_pt": round(box.width, 2), "height_pt": round(box.height, 2),
                       "text_characters": len(text)})
        target.close()
    (ROOT / "layout_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
