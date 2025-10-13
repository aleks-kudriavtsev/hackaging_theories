# Development Notes

This document captures additional guidance for contributors working on the Hackaging theories pipeline.

## Module map

- `src/theories_pipeline/literature.py` – Data access layer that turns JSON seed data or API payloads into `PaperMetadata` records.
- `src/theories_pipeline/theories.py` – Keyword-based classifier that converts `PaperMetadata` objects into `TheoryAssignment` scores.
- `src/theories_pipeline/extraction.py` – Implements the Q1–Q9 question set, along with `QuestionExtractor` helpers.
- `src/theories_pipeline/outputs.py` – Serialises `PaperMetadata`, `TheoryAssignment`, and `QuestionAnswer` objects to CSV.

## Local testing tips

1. Run `pytest` from the repository root; the suite uses mocked file data and does not require network connectivity.
2. Use `python scripts/collect_theories.py "activity engagement"` for an end-to-end smoke test.
3. To experiment with alternative keyword strategies, modify `config/pipeline.yaml` and rerun the CLI script.
4. Adaptive query expansion is documented in [`docs/query_expansion.md`](./query_expansion.md); consult it when tuning GPT prompts or embedding settings.

For additional architectural diagrams or extended walkthroughs, submit an issue so the maintainers can expand the documentation set.
