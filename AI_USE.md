# Use of AI tools

**Tool:** Claude (Anthropic), through the Claude Code extension for VS Code.

## Ideation

The project question (reference-query domain mismatch in NNN) and the proposed
fix (BLIP-generated captions as the reference bank) came out of a discussion with
Claude, which was also used for the literature search. The instructor was informed
and approved this use.

## What the AI tool did

- Wrote most of the code in `src/`, `scripts/` and `tests/` from the experiment
  design, and tested each script on small inputs before the full runs.
- Helped set up the environment (`requirements.txt`, `.gitignore`) and download
  the public datasets.
- Ran part of the experiments (tuning, evaluation, BLIP captioning) at the
  student's request; the student re-ran tuning and evaluation and obtained
  identical numbers.
- Drafted documentation (`README.md`, this file, `REUSED_COMPONENTS.md`) and
  helped check the paper's protocol details.

## What the student did

Chose the topic and made the design decisions (tuning set, bank sampling unit,
BLIP decoding, arms and seeds), ran the experiments, read the papers, interpreted the
results and wrote the report.
