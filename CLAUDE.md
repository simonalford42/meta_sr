# Project Notes for Claude

## Default LLM Model
Always use `openai/gpt-5-mini` as the default model for LLM calls in this codebase (not gpt-4o-mini).

## Committing code
This is a research project with no other collaborators, so committing is mainly useful for (1) saving work in case of catastrophic data loss, (2) going to previous versions if a major bug/regression occurs, to figure out what caused it. Please make commits and push them to github accordingly.

## submitting slurm jobs
Even if I gave you full permissions with --dangerously-skip-permissions, please ask for permission before running anything that would submit SLURM jobs.

## Do not use the `rm` command
Instead, move things to the trash ~/trash/
