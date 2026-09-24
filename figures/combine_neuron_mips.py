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

FIGURES = Path(__file__).resolve().parent


def main():
    paths = [FIGURES / 'neuronbench_uninformative_all_fits.pdf',
             FIGURES / 'mips_task_checkmarks.pdf']
    with pymupdf.open(paths[0]) as neuron, pymupdf.open(paths[1]) as mips:
        assert len(neuron) == len(mips) == 1, 'Expected single-page figures'
        height = neuron[0].rect.height
        widths = [doc[0].rect.width * height / doc[0].rect.height
                  for doc in (neuron, mips)]
        gap = 14  # PDF points; scale both panels uniformly to equal height.
        with pymupdf.open() as output:
            page = output.new_page(width=sum(widths) + gap, height=height)
            page.show_pdf_page(pymupdf.Rect(0, 0, widths[0], height), neuron, 0)
            page.show_pdf_page(pymupdf.Rect(widths[0] + gap, 0,
                                           sum(widths) + gap, height), mips, 0)
            output.set_metadata({'title': 'NeuronBench and MIPS',
                                 'subject': 'NeuronBench left; MIPS right'})
            target = FIGURES / 'neuron_mips_combined.pdf'
            output.save(target, garbage=4, deflate=True)
    print(target)


if __name__ == '__main__':
    main()
