# Aging Theory Literature Pipeline

This repository bundles a lightweight, four-stage workflow for harvesting PubMed
reviews about aging theories, curating the relevant subset with OpenAI, pulling
open-access full texts, and extracting the theories they discuss. Each stage is
implemented as a standalone script so you can inspect or extend the behaviour in
isolation, and a convenience runner (`scripts/run_full_pipeline.py`) chains them
into a single command for day-to-day use.

> **Note**  
> The project ships without vendored credentials. Export the API keys you are
> authorised to use before running the scripts. Example placeholders are shown
> below—replace them with your own secrets.

The pipeline focuses on three pillars of the challenge:

1. **Collection** – `src/theories_pipeline/literature.py` loads deterministic
   seed metadata for testing and can be extended with API providers (e.g.,
   OpenAlex, CrossRef, bioRxiv, medRxiv) for production runs.
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

## Aging theory review bootstrap pipeline

The repository now includes a four-stage bootstrap that focuses specifically on
aging-theory review articles retrieved from PubMed. The helper scripts live in
`scripts/step[1-4]_*.py`, and a convenience orchestrator
(`scripts/run_pipeline.py`) runs the entire sequence end-to-end.

### 1. Configure credentials

Export the API keys used throughout the pipeline. Only the OpenAI key is
strictly required, but providing your PubMed key and contact metadata will keep
the requests within NCBI's polite-use guidelines.

```bash
export PUBMED_API_KEY="your-ncbi-key"        # optional but recommended
export PUBMED_TOOL="your-app-name"           # optional tool identifier for NCBI
export PUBMED_EMAIL="you@example.com"        # optional contact email for NCBI
export OPENAI_API_KEY="sk-your-openai-token" # required for steps 2 and 4
```

### 2. Run the orchestrated pipeline

```bash
python scripts/run_pipeline.py --workdir data/pipeline
```

By default this performs:

1. `step1_pubmed_search.py` – exhaustive PubMed search for review articles
   matching the query `(aging theory OR ageing theory OR theories of aging)` in
   titles/abstracts, saving metadata to `start_reviews.json`.
2. `step2_filter_reviews.py` – OpenAI-powered screening of each record's title
   and abstract to discard off-topic material, storing decisions in
   `filtered_reviews.json`.
3. `step3_fetch_fulltext.py` – PMC full-text enrichment when an article has a
   matching PubMed Central entry, preserving paragraph boundaries so downstream
   models see intact sentences.
4. `step4_extract_theories.py` – LLM-based extraction of aging theories from the
   (full) texts, plus aggregation of unique theory names in
   `aging_theories.json`.

The orchestrator skips steps whose outputs already exist unless `--force` is
supplied. Use `--query`, `--filter-model`, `--theory-model`, or
`--max-chars` to customise individual stages.

### 3. Run stages individually (optional)

Each script doubles as a standalone CLI should you need to debug or tweak a
particular step:

```bash
# Step 1 – PubMed search
python scripts/step1_pubmed_search.py --output data/pipeline/start_reviews.json

# Step 2 – LLM relevance filtering
python scripts/step2_filter_reviews.py --input data/pipeline/start_reviews.json --output data/pipeline/filtered_reviews.json

# Step 3 – Full-text enrichment
python scripts/step3_fetch_fulltext.py --input data/pipeline/filtered_reviews.json --output data/pipeline/filtered_reviews_fulltext.json

# Step 4 – Theory extraction
python scripts/step4_extract_theories.py --input data/pipeline/filtered_reviews_fulltext.json --output data/pipeline/aging_theories.json
```

All scripts validate their inputs and surface helpful error messages when the
remote APIs reject a query or return malformed data, making it easier to spot
credential issues or intermittent network problems.

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
pip install --upgrade pip
```

The pipeline only relies on the Python standard library, so no additional
packages are required unless you plan to build optional extensions.

### 1. Configure credentials

```bash
export PUBMED_API_KEY="your-ncbi-eutilities-key"
export OPENAI_API_KEY="your-openai-key"
```

Providing a contact identity to PubMed is recommended by NCBI:

```bash
export PUBMED_TOOL="your-tool-name"
export PUBMED_EMAIL="maintainer@example.org"
```

### 2. Run the full pipeline

The helper script orchestrates the four stages and stores intermediate JSON
artefacts in `data/pipeline/` by default:

```bash
python scripts/run_full_pipeline.py \
  --query '(("aging theory"[TIAB] OR "ageing theory"[TIAB]) AND review[PTYP])' \
  --output-dir data/pipeline
```

Override `--filter-model`, `--extract-model`, or the request delays if you need
custom OpenAI models or throttling for quota management. When the command
completes successfully you should see four files:

- `start_reviews.json` – raw PubMed review metadata
- `filtered_reviews.json` – records that passed the LLM relevance filter
- `filtered_reviews_fulltext.json` – filtered records augmented with PubMed
  Central full texts where available
- `aging_theories.json` – the final catalogue with extracted theories

## Running stages manually

Each script accepts CLI flags so you can plug in alternate inputs/outputs or run
only part of the workflow while debugging. They can be executed independently:

```bash
# Step 1 — PubMed review search
python scripts/step1_pubmed_search.py \
  --output data/pipeline/start_reviews.json

# Step 2 — LLM-based relevance filtering
python scripts/step2_filter_reviews.py \
  --input data/pipeline/start_reviews.json \
  --output data/pipeline/filtered_reviews.json \
  --model gpt-4o-mini

# Step 3 — PubMed Central full-text enrichment
python scripts/step3_fetch_fulltext.py \
  --input data/pipeline/filtered_reviews.json \
  --output data/pipeline/filtered_reviews_fulltext.json

# Step 4 — Theory extraction from curated corpus
python scripts/step4_extract_theories.py \
  --input data/pipeline/filtered_reviews_fulltext.json \
  --output data/pipeline/aging_theories.json \
  --model gpt-4o-mini
```

All scripts share the following expectations:

- When an input path is supplied it must point to a UTF-8 encoded JSON file that
  matches the schema produced by the preceding step.
- `OPENAI_API_KEY` is required for steps 2 and 4; `PUBMED_API_KEY` is optional
  but highly recommended for steps 1 and 3.
- Network failures and HTTP errors are surfaced directly so you can retry or
  adjust rate limits as needed.

## Troubleshooting tips

- **PubMed rejects the query.** The collector validates the ESearch response and
  raises a helpful error message if NCBI reports query syntax issues. Double
  check the field tags in `--query` and ensure parentheses are balanced.
- **No PMC full text available.** Articles without a PubMed Central ID will have
  `"full_text": null` in `filtered_reviews_fulltext.json`. Supplement these
  manually if you have alternate access.
- **LLM quota exceeded.** Reduce `--filter-delay` / `--extract-delay` or run in
  smaller batches by splitting the input file before invoking the scripts.

## Repository layout

```
.
├── scripts/                  # Command-line entry points for each pipeline step
├── data/                     # Default output directory for pipeline artefacts
├── src/                      # Original hackathon toolkit (not required here)
└── tests/                    # Legacy pytest suite
```

Feel free to adapt the pipeline to ingest additional data sources (e.g.,
OpenAlex, Google Scholar, Sci-Hub mirrors). The modular structure is intended to
make it easy to slot in new providers or analysis stages.
