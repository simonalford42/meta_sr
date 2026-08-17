#!/usr/bin/env bash
set -euo pipefail

# NeuronBench currently declares Python >=3.11, while meta_sr is deliberately
# pinned to Python 3.10.  The deterministic API used by this demo is compatible
# with 3.10 (and is covered by the demo's validation command), so install the
# exact reviewed commit without changing the research environment.
NEURONBENCH_COMMIT="c354622458c460b419cab821d482c879f0578377"

conda run -n meta_sr python -m pip install \
  --no-deps \
  --ignore-requires-python \
  "git+https://github.com/murphyk/neuronbench.git@${NEURONBENCH_COMMIT}"

conda run -n meta_sr python -c \
  "import neuronbench, neuronbench.worlds as w; print(neuronbench.__file__); print(sorted(w.WORLDS))"
