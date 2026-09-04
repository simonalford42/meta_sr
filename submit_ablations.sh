#!/usr/bin/env bash

# 9/4/26

# Standard; also medium2, task population, 3-to-10 reeval, and zero-generation simplify cooldown.
chain_a=$(sbatch --parsable -J standaard run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2)

# Operator ablations.
chain_b=$(sbatch --parsable -J no-mut run.sh evolve_pysr.py --operator-type survival,selection,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2)
# sbatch -J no-loss run.sh evolve_pysr.py --operator-type mutation,survival,selection --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
# sbatch -J no-select run.sh evolve_pysr.py --operator-type mutation,survival,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
# sbatch -J no-survive run.sh evolve_pysr.py --operator-type mutation,selection,loss --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
# sbatch -J no-data-mut run.sh evolve_pysr.py --operator-type all --no-data-aware-mutations --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2

# Meta-mutation ablations.
# sbatch -J no-explore run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode explore --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J no-refine run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode refine --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2)
# sbatch -J no-cross run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode crossover --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
# sbatch -J no-simplify run.sh evolve_pysr.py --operator-type all --exclude-mutation-mode simplify --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2

# Simplify-cooldown
# sbatch -J cooldown15 run.sh evolve_pysr.py --operator-type all --simplify-cooldown 15 --population-type task --generations 25 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2

# Prompt-feedback ablations.
# sbatch -J no-feedback run.sh evolve_pysr.py --operator-type all --exec-feedback-n 0 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J uninfo-no-fb run.sh evolve_pysr.py --operator-type all --uninformative-prompts --exec-feedback-n 0 --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2)

# Reevaluation ablations; the standard condition is 3-to-10 population reeval.
chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J nruns1 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 1 --reeval none --models medium2)
# sbatch -J nruns3 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval none --models medium2
# sbatch -J nruns10 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 10 --reeval none --models medium2
# sbatch -J reeval1to3 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 1 --reeval population --n-reevals 3 --models medium2
# sbatch -J reeval-dyn run.sh evolve_pysr.py --operator-type all --population-type topk --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval TTTS --reeval-budget 20 --models medium2

# LLM ensemble quality; cheap2 is the available small2-equivalent preset, and standard is medium2.
chain_b=$(sbatch --parsable --dependency=afterany:"$chain_b" -J llm-small2 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models cheap2)
chain_a=$(sbatch --parsable --dependency=afterany:"$chain_a" -J llm-best2 run.sh evolve_pysr.py --operator-type all --population-type task --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models best2)

# Population selection; the standard condition uses task-diverse selection.
# sbatch -J pop-topk run.sh evolve_pysr.py --operator-type all --population-type topk --generations 10 --population 10 --offspring 10 --n-runs 3 --reeval population --n-reevals 10 --models medium2
