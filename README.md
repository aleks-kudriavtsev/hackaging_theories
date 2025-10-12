# Hackaging Theories Pipeline

The Hackaging challenge asks teams to map the gerontological theory landscape by
collecting literature, tagging each paper with the theories it engages, and
answering a standard set of nine analytical questions (Q1–Q9). This repository
packages that workflow into a reproducible Python toolkit so contributors can
run the end-to-end pipeline, extend individual modules, or plug the outputs into
their own analysis stack.

## Challenge overview

The pipeline focuses on three pillars of the challenge:

1. **Collection** – `src/theories_pipeline/literature.py` loads deterministic
   seed metadata for testing and can be extended with API providers (e.g.,
   OpenAlex, CrossRef) for production runs.
2. **Theory classification** – `src/theories_pipeline/theories.py` implements a
   transparent keyword matcher that scores how strongly each paper aligns with
   known theories. Teams can substitute this module with more advanced models
   while preserving the same interface.
3. **Question extraction** – `src/theories_pipeline/extraction.py` automates the
   Hackaging Q1–Q9 prompts so findings, methods, and limitations are captured in
   structured CSV form.

Sample inputs and outputs in `data/examples/` illustrate the expected artefacts
that the Hackaging organisers require for leaderboard submissions.

## Repository structure

```
.
├── config/                 # YAML/JSON pipeline configuration templates
├── data/examples/          # Seed dataset and sample CSV outputs
├── docs/                   # Supplementary developer documentation
├── scripts/                # Command line entry points for collection/analysis
├── src/theories_pipeline/  # Core Python package with reusable modules
└── tests/                  # Automated pytest suite with mocked inputs
```

See [`docs/development.md`](docs/development.md) for an expanded module map and
local testing tips.

## Setup instructions

### Prerequisites

- Python 3.10 or newer
- `pip` 22+ (for modern dependency resolution)

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies

The project only depends on the standard library plus a small set of helper
packages used for configuration parsing and tests:

```bash
pip install pyyaml pytest
```

Install any additional providers (e.g., HTTP clients) required for your custom
retrieval strategies in the same environment.

### Environment variables

External API keys are stored in the configuration under `api_keys`, but you
should prefer environment variables during development to avoid committing
secrets. The recommended variables are:

- `OPENALEX_API_KEY`
- `CROSSREF_API_KEY`

The bundled CLI scripts do not read these variables directly; instead, reference
them when generating your own config (e.g., with `envsubst`) or when extending
`LiteratureRetriever` with real API clients.

## Running the pipelines

### Collect theories and initial Q1–Q9 answers

```bash
python scripts/collect_theories.py "activity engagement" --config config/pipeline.yaml
```

This command

1. Loads seed papers from `config/pipeline.yaml` via
   `LiteratureRetriever.search()`.
2. Classifies each paper with `TheoryClassifier.from_config()`.
3. Extracts question responses using `QuestionExtractor.extract()`.
4. Writes CSVs using the helpers in `src/theories_pipeline/outputs.py`.

The resulting CSVs land in `data/examples/` by default.

### Refresh question answers and generate a summary cache

```bash
python scripts/analyze_papers.py --config config/pipeline.yaml
```

This utility reads an existing papers CSV (or falls back to the seed data),
recomputes theory counts, re-exports Q1–Q9 answers, and emits
`data/cache/analysis_summary.json` for downstream dashboards.

## CSV schemas

The exporter utilities in `src/theories_pipeline/outputs.py` enforce consistent
headers for every CSV artefact. Expect the following columns:

| File | Function | Columns |
| --- | --- | --- |
| `data/examples/papers.csv` | `export_papers` | `identifier`, `title`, `authors`, `abstract`, `source`, `year`, `doi` |
| `data/examples/theories.csv` | `export_theories` | `paper_id`, `theory`, `score` (stringified to three decimal places) |
| `data/examples/questions.csv` | `export_question_answers` | `paper_id`, `question_id`, `question`, `answer` |

Each row in the questions export corresponds to one of the nine constants in
`src/theories_pipeline/extraction.py::QUESTIONS`, ensuring the Q1–Q9 prompts stay
aligned across runs.

## Contributor guide

We welcome contributions that improve coverage, extraction accuracy, and data
quality. Before opening a pull request:

1. Format code with the default Python style (PEP 8 / `black` conventions are
   acceptable; avoid introducing new dependencies for styling).
2. Run the unit tests:
   ```bash
   pytest
   ```
3. Execute the collection script against the sample dataset to confirm the CSVs
   update without errors:
   ```bash
   python scripts/collect_theories.py "activity engagement"
   ```
4. Document notable changes in `docs/` if they alter module behaviour or the
   data contract. Reference the new materials from this README to keep the entry
   point up to date.

For more detailed development notes, see
[`docs/development.md`](docs/development.md). If you create additional deep dives
(e.g., architecture diagrams or benchmarking results), place them under `docs/`
and link them from this section so future contributors can discover them easily.
