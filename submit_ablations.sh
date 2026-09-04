#!/usr/bin/env bash

# 9/4/26

# Standard; also best2, task population, 3-to-10 reeval, and zero-generation simplify cooldown.
sbatch -J standaard run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2

# Operator ablations.
sbatch -J abl-no-mut run.sh evolve_pysr.py --operator-type survival,selection,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-loss run.sh evolve_pysr.py --operator-type mutation,survival,selection --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-select run.sh evolve_pysr.py --operator-type mutation,survival,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-survive run.sh evolve_pysr.py --operator-type mutation,selection,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-data-mut run.sh evolve_pysr.py --operator-type all --no-data-aware-mutations --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2

# Meta-mutation ablations.
sbatch -J abl-no-explore run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode explore --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-refine run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode refine --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-cross run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode crossover --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-no-simplify run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode simplify --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2

# Simplify-cooldown sweep; the standard condition is cooldown 0.
sbatch -J abl-cooldown2 run.sh evolve_pysr.py --operator-type all --simplify-cooldown 2 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-cooldown5 run.sh evolve_pysr.py --operator-type all --simplify-cooldown 5 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2

# Prompt-feedback ablations.
sbatch -J abl-no-feedback run.sh evolve_pysr.py --operator-type all --exec-feedback-n 0 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
sbatch -J abl-uninfo-no-fb run.sh evolve_pysr.py --operator-type all --uninformative-prompts --exec-feedback-n 0 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2

# Reevaluation ablations; the standard condition is 3-to-10 population reeval.
sbatch -J abl-nruns1 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 1 --reeval none --models best2
sbatch -J abl-nruns3 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval none --models best2
sbatch -J abl-nruns10 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 10 --reeval none --models best2
sbatch -J abl-reeval1to3 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 1 --reeval population --n-reevals 3 --models best2
sbatch -J abl-reeval-dyn run.sh evolve_pysr.py --operator-type all --population-type topk --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval TTTS-dynamic --reeval-budget 20 --models best2

# LLM ensemble quality; cheap2 is the available small2-equivalent preset, and standard is best2.
sbatch -J abl-llm-small2 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models cheap2
sbatch -J abl-llm-medium2 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2

# Population selection; the standard condition uses task-diverse selection.
sbatch -J abl-pop-topk run.sh evolve_pysr.py --operator-type all --population-type topk --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2
