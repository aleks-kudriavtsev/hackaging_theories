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
   Hackaging Q1–Q9 prompts spanning biomarkers, mechanisms, interventions, and
   species-level comparisons so categorical outputs are captured in structured
   CSV form.

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

The project depends on a lightweight stack of HTTP and parsing helpers. Install
the required packages (and optional extras for PDF parsing) with:

```bash
pip install pyyaml pytest requests pdfminer.six
```

`requests` powers the provider clients, while `pdfminer.six` enables the PDF
fallback used when only a portable-document full text is available. You can skip
`pdfminer.six` if you plan to work solely with HTML or plain-text sources.

### Environment variables

External API keys are declared in the configuration under `api_keys`, and the
CLI entry points resolve them via environment variables, `.env` files, or explicit
CLI overrides. The most common credentials are summarised below:

| Service | Purpose | Environment variable | Notes |
| --- | --- | --- | --- |
| OpenAlex | High-quality bibliographic metadata and OA full texts | `OPENALEX_API_KEY` | Required for high-throughput access; register via [OpenAlex dashboard](https://docs.openalex.org/how-to-use-the-api/api-overview). |
| Crossref | Supplementary metadata / DOI lookups | `CROSSREF_API_KEY` | Crossref expects a valid contact email; use `--crossref-contact` to override per run. |
| PubMed E-Utilities | Biomedical abstracts and some full texts | `PUBMED_API_KEY` | Obtain from NCBI (see [E-Utilities guide](https://www.ncbi.nlm.nih.gov/books/NBK25501/)). |
| OpenAI | LLM-assisted filtering, ontology generation, and Q1–Q9 answers | `OPENAI_API_KEY` | Supports any compatible Chat Completions model (e.g., `gpt-4o-mini`). |
| SerpAPI (Google Scholar) | Additional review discovery | `SERPAPI_KEY` | Optional; enables the Google Scholar provider. |
| Semantic Scholar | Metadata and abstracts from the Semantic Scholar corpus | `SEMANTIC_SCHOLAR_API_KEY` | Optional but recommended for expanded coverage. |

Set the variables once per shell session, or pass overrides such as
`--openalex-api-key` / `--serpapi-key` directly to the collection CLI. See
`src/theories_pipeline/config_utils.py` for the supported descriptor syntax and
`config/pipeline.yaml` for annotated examples.

## Quickstart: generate an ontology from reviews (no seed config required)

The quickstart workflow mirrors the four steps you outlined:

1. **Harvest candidate reviews.** The retriever queries OpenAlex, Crossref,
   PubMed, Semantic Scholar, and (optionally) Google Scholar via SerpAPI to
   gather titles and abstracts for highly relevant "aging theory" papers.
2. **Filter with an LLM.** `RelevanceFilter` uses optional GPT assistance to
   discard off-topic hits before bootstrapping an ontology.
3. **Generate an ontology.** `review_bootstrap.bootstrap_ontology()` extracts
   theory labels from the accepted review abstracts (plus optional GPT
   suggestions) and persists the generated hierarchy to
   `data/cache/ontologies/<query>.json`.
4. **Populate nodes.** The main collection loop reuses the generated ontology to
   retrieve and analyse the broader corpus, automatically exporting papers,
   theory assignments, and Q1–Q9 answers.

### Step-by-step command sequence

```bash
export OPENALEX_API_KEY="..."              # required for OpenAlex
export CROSSREF_API_KEY="you@example.com"  # Crossref contact email
export PUBMED_API_KEY="..."                # optional but recommended
export OPENAI_API_KEY="..."                # enables GPT filtering/ontology
# Optional extras
export SERPAPI_KEY="..."                   # enable Google Scholar provider
export SEMANTIC_SCHOLAR_API_KEY="..."      # Semantic Scholar provider

python scripts/collect_theories.py "aging theory" \
  --config config/pipeline.yaml \
  --quickstart \
  --target-count 300 \
  --llm-model gpt-4o-mini \
  --parallel-fetch 6 \
  --classification-workers 6
```

What happens during the run:

- A review snapshot is saved to `data/cache/ontologies/aging-theory.json` with
  filter decisions and the generated ontology tree.
- Papers that pass the relevance filter are kept in memory for ontology
  generation and seeding; you can inspect them in the snapshot file.
- The collection stage then fills each ontology node using the same providers.
- Outputs land in `data/examples/papers.csv`, `theories.csv`, and
  `questions.csv` unless you override `config.outputs`.

> **Note on full texts:** The retriever only uses OpenAlex, PubMed, Semantic
> Scholar, and other legally licensed sources for obtaining full texts. If only a
> PDF is available, the optional `pdfminer.six` dependency converts it into text
> for downstream analysis. Services such as Sci-Hub or Anna's Archive are not
> integrated because they distribute copyrighted content without permission.

### Iterating on the generated ontology

- The cached ontology file is plain JSON—edit it manually or run another
  quickstart with different parameters to compare hierarchies.
- Re-running `collect_theories.py` with `--quickstart` reuses the cached
  ontology. Delete the corresponding snapshot file if you want to regenerate the
  hierarchy from scratch or adjust `--target-count` to rebalance node targets.
- To refresh Q1–Q9 answers or reclassify papers without additional retrieval,
  run `python scripts/analyze_papers.py --config config/pipeline.yaml`.

## Running the pipelines with a curated ontology

If you prefer the original config-driven workflow (for regression testing or
benchmarking), disable quickstart and rely on the static `corpus.targets`
section:

```bash
python scripts/collect_theories.py "activity engagement" --config config/pipeline.yaml
```

The command

1. Loads seed papers from `config/pipeline.yaml` via `LiteratureRetriever`.
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
| `data/examples/questions.csv` | `export_question_answers` | `paper_id`, `question_id`, `question`, `answer`, `confidence`, `evidence` |

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
