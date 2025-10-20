# Aging Theory Literature Pipeline

This repository bundles an end-to-end workflow for harvesting PubMed reviews
about aging theories, curating the relevant subset with OpenAI, pulling
open-access full texts, extracting the theories they discuss, and then rolling
those ontology outputs straight into the general-literature collector and
classifier. Each stage is implemented as a standalone script so you can inspect
or extend the behaviour in isolation, and a convenience runner
(`scripts/run_pipeline.py`) chains them into a single command for day-to-day
use. The legacy `scripts/run_full_pipeline.py` module now forwards to the same
entrypoint so existing automation keeps working.

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

The repository now includes a multi-stage bootstrap that focuses specifically on
aging-theory review articles retrieved from PubMed and continues through to
ontology-driven retrieval/classification. The helper scripts live in
`scripts/step[1-5]_*.py`, and a convenience orchestrator
(`scripts/run_pipeline.py`) runs the entire sequence end-to-end while merging
records from PubMed, OpenAlex, optional Google Scholar collectors, and the
follow-on literature crawl.

### 1. Configure credentials

Export the API keys used throughout the pipeline. Only the OpenAI key is
strictly required, but providing your PubMed key and contact metadata will keep
the requests within NCBI's polite-use guidelines.

```bash
export PUBMED_API_KEY="your-ncbi-key"        # optional but recommended
export PUBMED_TOOL="your-app-name"           # optional tool identifier for NCBI
export PUBMED_EMAIL="you@example.com"        # optional contact email for NCBI
export OPENAI_API_KEY="sk-your-openai-token" # required for steps 2, 4, and 5
```

### 2. Run the orchestrated pipeline

```bash
python scripts/run_pipeline.py --workdir data/pipeline
```

By default this performs:

1. `step1_pubmed_search.py` – exhaustive PubMed search for review articles
   matching the query `(aging theory OR ageing theory OR theories of aging)` in
   titles/abstracts, saving metadata to `start_reviews_pubmed.json`.
2. `step1b_openalex_search.py` – OpenAlex metadata harvest for the same keyword
   set, stored in `start_reviews_openalex.json`.
3. Optional Google Scholar collector (`step1c_*`) when present.
4. Deduplication and merging of the provider outputs into a single
   `start_reviews.json` file keyed on DOI/PMID/OpenAlex IDs.
5. `step2_filter_reviews.py` – OpenAI-powered screening of each record's title
   and abstract to discard off-topic material, storing decisions in
   `filtered_reviews.json`. The script now auto-detects when to spawn multiple
   worker processes (defaulting to the machine's CPU count for 100+ items) and
   streams per-process progress logs. The default `gpt-5-nano` model balances
   quality with the target ~$10 per million articles budget while comfortably
   handling batched abstracts.
6. `step3_fetch_fulltext.py` – PMC full-text enrichment when an article has a
   matching PubMed Central entry, preserving paragraph boundaries so downstream
   models see intact sentences.
7. `step4_extract_theories.py` – LLM-based extraction of aging theories from the
   (full) texts, plus aggregation of unique theory names in
   `aging_theories.json`. Long reviews are split into overlapping prompt
   windows controlled by `--chunk-chars` and `--chunk-overlap`, and theory names
   from every chunk are deduplicated before being stored on the record. The
   worker count scales with the CPU total for queues above 100 items and can be
   overridden with `--processes` when you need explicit control. Theory
   extraction defaults to `gpt-5-nano`, and the optional hypothesis review stage
   runs on `gpt-4.1-nano` so both passes stay accurate within the same budget
   envelope.
8. `step5_generate_ontology.py` – LLM-assisted grouping of the extracted
   theories into multi-level ontology clusters saved as `aging_ontology.json`.
   Ontology synthesis defaults to `gpt-5-mini`, which provides extra synthesis
   capacity without exceeding the approximate $10 per million articles spend.
9. `collect_theories.py` – parses the reconciled ontology, turns each group and
   theory into transient retrieval targets, expands their suggested queries, and
   runs the literature collector/classifier using an in-memory
   `config/pipeline.yaml` snapshot. The exported papers, assignments, and
   question answers mirror the standalone collector's CSV outputs and are saved
   under `<workdir>/collector/` by default (alongside the collector cache).

The orchestrator skips steps whose outputs already exist unless `--force` is
supplied. Use `--query`, `--collector-query` (or the alias `--base-query`),
`--filter-model`, `--theory-model` (or the legacy
`--extract-model` alias), `--hypothesis-review-model`, the chunking options
(`--chunk-chars`, `--chunk-overlap`), or the ontology arguments
(`--ontology-model`, `--ontology-top-n`, `--ontology-examples`) to customise
individual stages. Override the defaults when you need to trade context for cost
(e.g., swapping `gpt-5-nano` for an even cheaper tier on short abstracts) or
when your account exposes alternative model families.

Collector-specific flags such as `--limit` and `--state-dir` now propagate
through the unified runner so you can cap exported papers or pin the retrieval
cache without switching to the standalone `collect_theories.py` entry point.

### 2a. Loop the full review → ontology → enrichment cycle

Operators who want the freshly generated ontology to immediately seed the
general-literature crawl can rely on the new convenience wrapper:

```bash
python scripts/run_full_cycle.py --workdir data/pipeline
```

The command first invokes `run_pipeline.py` to rebuild steps 1–5 inside the
chosen work directory. It then parses `aging_ontology.json`, converts the final
theory groups into the `corpus.targets` structure used by
`collect_theories.py`, merges any `suggested_queries`, and runs the retrieval
and promotion phase. The collector persists its state, CSV exports, and runtime
ontology updates under the same workdir, so a single folder captures both the
review bootstrap artefacts and the subsequent enrichment results. Pass through
options such as `--limit`, `--providers`, or `--no-resume` when you need to
fine-tune the second phase.

**Recommended model mix.** For a balanced run that keeps the end-to-end spend
close to $10 per million processed articles:

- Stage 2 (relevance filtering) – `gpt-5-nano` for fast, low-cost classification.
- Stage 4 (theory extraction) – `gpt-5-nano` for consistent naming while staying
  within the budget.
- Hypothesis review (post-extraction audit) – `gpt-4.1-nano` to add lightweight
  structured reasoning before ontology grouping.
- Stage 5 (ontology generation) – `gpt-5-mini` for higher-quality synthesis when
  reconciling theories into a shared ontology.

Operators should override these defaults if they have tighter budgets, require
larger context windows, or prefer vendor-specific alternatives.

### 3. Run stages individually (optional)

Each script doubles as a standalone CLI should you need to debug or tweak a
particular step:

```bash
# Step 1 – PubMed search
python scripts/step1_pubmed_search.py --output data/pipeline/start_reviews_pubmed.json

# Step 1b – OpenAlex search
python scripts/step1b_openalex_search.py --output data/pipeline/start_reviews_openalex.json

# Step 1c – Optional Google Scholar search (only if the helper exists)
python scripts/step1c_google_scholar.py --output data/pipeline/start_reviews_google_scholar.json

# Merge provider outputs before filtering (handled automatically by run_pipeline.py)
# Copy or concatenate the collected JSON into data/pipeline/start_reviews.json

# Step 2 – LLM relevance filtering
python scripts/step2_filter_reviews.py --input data/pipeline/start_reviews.json --output data/pipeline/filtered_reviews.json --processes 4

# Step 3 – Full-text enrichment
python scripts/step3_fetch_fulltext.py --input data/pipeline/filtered_reviews.json --output data/pipeline/filtered_reviews_fulltext.json

# Step 4 – Theory extraction
python scripts/step4_extract_theories.py --input data/pipeline/filtered_reviews_fulltext.json --output data/pipeline/aging_theories.json --chunk-chars 12000 --chunk-overlap 1000 --processes 4

# Step 5 – Ontology generation
python scripts/step5_generate_ontology.py --input data/pipeline/aging_theories.json --output data/pipeline/aging_ontology.json
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

The helper script orchestrates collection, deduplication, filtering, full-text
enrichment, theory extraction, and ontology building. Intermediate JSON
artefacts are stored in `data/pipeline/` by default:

```bash
python scripts/run_pipeline.py \
  --query '(("aging theory"[TIAB] OR "ageing theory"[TIAB]) AND review[PTYP])' \
  --workdir data/pipeline
```

Override `--filter-model`, `--theory-model` (or the compatibility flag
`--extract-model`), or the request delays if you need custom OpenAI models or
throttling for quota management. The filtering stage also accepts
`--batch-size` so multiple abstracts can be screened in a single request.
Parallelism can be tuned via `--processes` (defaults to the CPU count for large
inputs) when you need to split long queues across worker processes. The
filtering and theory extraction stages both honour this flag. Start with 5–10
items for GPT-4o/GPT-4o mini tiers (roughly 3–5k prompt tokens) and lower the
value if your abstracts are unusually long or you are using a model with a
smaller context window. When the command completes successfully you should see
the merged metadata (`start_reviews.json`) alongside the filtered, full-text,
theory, and ontology artefacts in `data/pipeline/`.

## Running stages manually

Each script accepts CLI flags so you can plug in alternate inputs/outputs or run
only part of the workflow while debugging. They can be executed independently:

```bash
# Step 1 — PubMed review search
python scripts/step1_pubmed_search.py \
  --output data/pipeline/start_reviews_pubmed.json

# Step 1b — OpenAlex review search
python scripts/step1b_openalex_search.py \
  --output data/pipeline/start_reviews_openalex.json

# Step 1c — Optional Google Scholar review search (if available)
python scripts/step1c_google_scholar.py \
  --output data/pipeline/start_reviews_google_scholar.json

# Merge the provider outputs into start_reviews.json before continuing

# Step 2 — LLM-based relevance filtering
python scripts/step2_filter_reviews.py \
  --input data/pipeline/start_reviews.json \
  --output data/pipeline/filtered_reviews.json \
  --model gpt-5-nano

# Step 3 — PubMed Central full-text enrichment
python scripts/step3_fetch_fulltext.py \
  --input data/pipeline/filtered_reviews.json \
  --output data/pipeline/filtered_reviews_fulltext.json

# Step 4 — Theory extraction from curated corpus
python scripts/step4_extract_theories.py \
  --input data/pipeline/filtered_reviews_fulltext.json \
  --output data/pipeline/aging_theories.json \
  --model gpt-5-nano

# Step 5 — Ontology generation
python scripts/step5_generate_ontology.py \
  --input data/pipeline/aging_theories.json \
  --output data/pipeline/aging_ontology.json \
  --model gpt-5-mini
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
