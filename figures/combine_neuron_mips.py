#!/usr/bin/env python3
"""Place the NeuronBench and MIPS PDFs side by side, retaining vector artwork.

Regenerate inputs as needed:
    python figures/plot_neuronbench_uninformative.py
    python figures/plot_mips_task_checkmarks.py
Then run:
    python figures/combine_neuron_mips.py
"""
from pathlib import Path

import pymupdf
from matplotlib.font_manager import FontProperties, findfont

FIGURES = Path(__file__).resolve().parent


def main():
    paths = [FIGURES / 'neuronbench_uninformative_all_fits.pdf',
             FIGURES / 'mips_task_checkmarks.pdf']
    with pymupdf.open(paths[0]) as neuron, pymupdf.open(paths[1]) as mips:
        assert len(neuron) == len(mips) == 1, 'Expected single-page figures'
        height = neuron[0].rect.height
        widths = [doc[0].rect.width * height / doc[0].rect.height
                  for doc in (neuron, mips)]
        gap = 20  # PDF points; scale both panels uniformly to equal height.
        title_height = 28
        title_font = findfont(FontProperties(family='DejaVu Serif', weight='bold'))
        font = pymupdf.Font(fontfile=title_font)
        with pymupdf.open() as output:
            page = output.new_page(width=sum(widths) + gap, height=height + title_height)
            page.show_pdf_page(pymupdf.Rect(0, title_height, widths[0], height + title_height), neuron, 0)
            page.show_pdf_page(pymupdf.Rect(widths[0] + gap, title_height,
                                           sum(widths) + gap, height + title_height), mips, 0)
            page.insert_font(fontname='PanelTitle', fontfile=title_font)
            for label, center in [('NeuronBench', widths[0] / 2),
                                  ('MIPS', widths[0] + gap + widths[1] / 2)]:
                text_width = font.text_length(label, fontsize=14)
                page.insert_text((center - text_width / 2, 17), label,
                                 fontname='PanelTitle', fontsize=14)
            output.set_metadata({'title': 'NeuronBench and MIPS',
                                 'subject': 'NeuronBench left; MIPS right'})
            target = FIGURES / 'neuron_mips_combined.pdf'
            output.save(target, garbage=4, deflate=True)
    print(target)


if __name__ == '__main__':
    main()
