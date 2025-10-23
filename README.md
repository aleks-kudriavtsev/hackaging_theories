# Aging Theory Literature Pipeline

This repository implements a fully automated loop for harvesting review
articles about aging theories, distilling their claims into a navigable
ontology, and continuously expanding that ontology with new primary literature.
Every stage is available as a standalone script for transparency, and the
orchestrators in `scripts/run_pipeline.py` and `scripts/run_full_cycle.py` chain
them into end-to-end commands that scale to very large corpora.

## Quick start

1. **Create a virtual environment (optional but recommended).**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

   The default implementation only relies on the Python standard library. Install
   [`pyyaml`](https://pyyaml.org/) if you plan to load YAML collector configs and
   [`pdfminer.six`](https://github.com/pdfminer/pdfminer.six) if you want PDF
   text extraction during full-text harvesting.

2. **Export the credentials used by the collectors.**

   ```bash
   export OPENAI_API_KEY="sk-your-openai-token"      # required for steps 2, 4, 5, and classification
   export PUBMED_API_KEY="your-ncbi-key"            # optional but recommended for polite E-utilities use
   export PUBMED_TOOL="your-tool-name"              # optional NCBI contact metadata
   export PUBMED_EMAIL="maintainer@example.org"     # optional NCBI contact metadata
   ```

   Additional provider keys (OpenAlex, Crossref, SerpApi, Semantic Scholar,
   Sci-Hub mirrors, Anna's Archive) can be supplied through environment
   variables or the collector configuration when you enable those services.

3. **Run the staged pipeline.**

   ```bash
   python scripts/run_pipeline.py --workdir data/pipeline
   ```

   This command executes the six linked stages described below, writing all
   intermediate artefacts to `data/pipeline/`. Use `--force` to regenerate
   existing outputs, `--query` to tweak the PubMed search, and the options listed
   under [Parallelism & performance](#parallelism--performance) to tune large
   runs.

4. **Loop the full review → ontology → enrichment cycle.**

   When you want the freshly generated ontology to immediately seed the
   general-literature crawl, switch to the full-cycle wrapper:

   ```bash
   python scripts/run_full_cycle.py --workdir data/pipeline
   ```

   The wrapper reuses `run_pipeline` for the review bootstrap and then invokes
   `scripts/collect_theories.py` with ontology-derived targets. All CSV exports,
   caches, and the global rejection registry end up under `data/pipeline/` by
   default so subsequent runs resume automatically.

## Stage-by-stage execution

### Step 1A – PubMed review bootstrap (`scripts/step1_pubmed_search.py`)

**Canonical command**

```bash
python scripts/step1_pubmed_search.py \
    --output data/pipeline/start_reviews.json
```

**Key flags and inputs**

* `--query` overrides the default PubMed Title/Abstract search that narrows the
  corpus to review articles about aging theories. Use it to experiment with
  tighter date ranges or alternate vocabularies without editing code.
* `--output` controls where the metadata JSON is written. The default is
  `data/pipeline/start_reviews.json`, but you can point to
  `start_reviews_pubmed.json` when you want to keep PubMed output separate from
  OpenAlex before merging.
* Optional environment variables—`PUBMED_API_KEY`, `PUBMED_TOOL`, and
  `PUBMED_EMAIL`—raise rate limits and attach contact metadata to your E-utility
  calls.

**Parallelism and staging tips**

The script streams requests sequentially but respects the PubMed policies by
pausing ~0.34 seconds between batches; raise this delay manually if you see 429
errors. Re-running the command with the same `--output` simply overwrites the
JSON, so keep dated filenames when you want incremental checkpoints.

### Step 1B – OpenAlex enrichment (`scripts/step1b_openalex_search.py`)

**Canonical command**

```bash
python scripts/step1b_openalex_search.py \
    --output data/pipeline/start_reviews_openalex.json
```

**Key flags and inputs**

* `--terms` lists the default aging-theory phrases. Provide your own phrases or
  supply a file of synonyms to broaden discovery.
* `--per-page` (default 200) and `--filter` arguments mirror the OpenAlex REST
  API query language; they let you keep the response size manageable while
  chaining extra filters such as publication years.
* Network hygiene is tunable with `--request-interval`, `--max-attempts`, and
  `--retry-wait`. Each has an accompanying environment variable (`OPENALEX_*`)
  so you can centralise rate-limit settings when running from orchestrators.

**Parallelism and staging tips**

The collector deduplicates on the OpenAlex work identifier, which means you can
append additional `--terms` in follow-up runs and then merge JSON payloads by
hand or through `run_pipeline.py`. Keep the API key (`OPENALEX_API_KEY`) in your
environment to benefit from higher request quotas.

### Step 2 – LLM relevance filter (`scripts/step2_filter_reviews.py`)

**Canonical command**

```bash
python scripts/step2_filter_reviews.py \
    --input data/pipeline/start_reviews.json \
    --output data/pipeline/filtered_reviews.json
```

**Key flags and inputs**

* `--model` selects the OpenAI chat model (default `gpt-5-nano`); bump to
  `gpt-5-mini` for stricter judgements when cost allows.
* `--batch-size` and `--concurrency` trade off token use and latency by packing
  multiple abstracts per request and limiting simultaneous API calls.
* `--processes` lets you pin the number of worker processes; leaving it blank
  auto-scales based on queue size and CPU availability. The same value now acts
  as the default for the downstream extraction stage unless you override
  `--extraction-processes`.

**Parallelism and staging tips**

Set `OPENAI_API_KEY` in your shell before launching. The script fan-outs to as
many worker processes as needed (capped by CPU cores) while maintaining an
inter-request delay to respect OpenAI rate limits. For long queues, write to a
dated `--output` path so that you can resume from the last successful file if a
network failure stops the run.

### Step 2 – Filter + full-text enrichment (`scripts/step2_filter_reviews.py`)

**Canonical command**

```bash
python scripts/step2_filter_reviews.py \
    --input data/pipeline/start_reviews.json \
    --filtered-output data/pipeline/filtered_reviews.json \
    --output data/pipeline/aging_theories.json \
    --processes 4 --extraction-processes 4
```

**Key flags and inputs**

* `--model`, `--delay`, `--batch-size`, `--cache`, and `--processes` govern the
  initial relevance filter against titles/abstracts.
* `--fulltext-processes` and `--fulltext-concurrency` switch between process and
  thread workers for PMC/OpenAlex retrieval when you want to override the
  auto-scaling defaults. The same arguments are passed through to the
  standalone fetcher described in [Step 3](#step-3--full-text-retrieval--pdf-normalisation-scriptsstep3_fetch_fulltextpy).
* `--entrez-interval`, `--entrez-max-attempts`, `--entrez-retry-wait`, and
  `--entrez-batch-size` mirror the PubMed E-utilities knobs exposed in
  `step3_fetch_fulltext.py`; environment variables with the same names continue
  to work.
* `--extraction-model`, `--extraction-processes`, `--extraction-delay`,
  `--chunk-chars`, and `--chunk-overlap` configure the gpt-5-nano theory
  extraction pipeline that now runs immediately after full-text retrieval. When
  you omit `--extraction-processes`, the script reuses the `--processes` value
  (subject to CPU and workload limits) before falling back to auto-scaling.
* `--failures` writes unresolved PDF/PMC downloads to `<output>.failures.json`
  so you can retry them later without touching already-processed articles.

**Parallelism and staging tips**

  The integrated step streams each relevant review through three stages without
  round-tripping intermediate JSON: filtering, full-text harvesting (PMC first,
  then OpenAlex PDF parsing/OCR), and gpt-5-nano theory extraction. OpenAlex PDF
  links are downloaded automatically during this phase and normalised using the
  same pipeline exposed by `step3_fetch_fulltext.py`. The output is
  the enriched `aging_theories.json` bundle containing the per-article full text,
  raw `raw_theory_mentions`, and the canonical `theory_registry` consumed by step
  5. Optional scripts `step3_fetch_fulltext.py` and `step4_extract_theories.py`
  remain available when you need to diagnose a single stage in isolation.

### Step 3 – Full-text retrieval & PDF normalisation (`scripts/step3_fetch_fulltext.py`)

**Canonical command**

```bash
python scripts/step3_fetch_fulltext.py \
    --input data/pipeline/filtered_reviews.json \
    --output data/pipeline/filtered_reviews_fulltext.json
```

**Key flags and inputs**

* `--failures` writes unresolved downloads and OCR problems to a companion log
  (defaults to `<output>.failures.json`).
* `--processes` and `--concurrency` mirror the worker controls surfaced in
  step 2, letting you favour OS processes or threads when fetching PMC XML and
  OpenAlex PDFs at scale.
* `--entrez-interval`, `--entrez-max-attempts`, `--entrez-retry-wait`, and
  `--entrez-batch-size` carry over the PubMed rate-limit knobs; the same
  environment variables (`PUBMED_*`) apply when you run the script standalone.

**PDF processing behaviour and dependencies**

The fetcher always prefers PMC body text, but when only OpenAlex supplies a PDF
it downloads the file by default. Digital PDFs are parsed with
[`pdfminer.six`](https://github.com/pdfminer/pdfminer.six); scanned PDFs fall back
to OCR via [`pdf2image`](https://github.com/Belval/pdf2image) and
[`pytesseract`](https://github.com/tesseract-ocr/tesseract) when those packages
are installed. You will need system packages for Poppler (required by
`pdf2image`) and Tesseract OCR to enable the conversion pipeline. When the
dependencies are missing, the script records the failure reason in the
`pdf_processing` metadata so you can retry later.

> The combined [Step 2](#step-2--filter--full-text-enrichment-scriptsstep2_filter_reviewspy)
> workflow automatically invokes this PDF download/OCR routine, so you only need
> to run Step 3 directly when debugging or replaying a subset of records.

### Step 5 – Ontology synthesis (`scripts/step5_generate_ontology.py`)

**Canonical command**

```bash
python scripts/step5_generate_ontology.py \
    --input data/pipeline/aging_theories.json \
    --output data/pipeline/aging_ontology.json
```

**Key flags and inputs**

* `--model` (default `gpt-5-mini`) governs the ontology synthesis quality; pair
  it with `--top-n` to trim the registry summary when you want cheaper trial
  runs.
* `--processes`, `--chunk-size`, and `--examples-per-theory` shape how the
  canonical registry is split across worker prompts and how many representative
  titles accompany each theory.
* Use `--llm-response` to hydrate the reconciler from a saved model response and
  `--max-theories-per-group` to enforce balanced group sizes when rebuilding the
  ontology.
* When the input file lacks a `theory_registry`, the script can reconstruct it
  from checkpoint annotations using the fallback model specified by
  `--registry-model` and the timeout in `--registry-request-timeout`.

**Two-pass consolidation**

After the initial `generate_grouping` prompts finish, the script runs a
gpt-5-mini consolidation pass over the chunked group summaries. The assistant
reasons about which top-level groups should share a parent and emits a merge
plan without touching article assignments. The resulting hierarchy flows into
the refinement step and every decision—suggested merges, created parents,
skipped references—is captured in the output JSON under
`ontology.consolidated.metadata` and mirrored in the reconciliation notes. The
top-level `reconciliation_report.llm_passes` map snapshots both metadata blocks
so downstream tooling can audit the consolidation and refinement passes when
replaying cached responses.

**Parallelism and staging tips**

Set `OPENAI_API_KEY` unless you feed a cached `--llm-response`. Large corpora
benefit from multi-process mode, but keep an eye on token budgets—chunking keeps
individual prompts within the ~240k token safety margin enforced by the prompt
builder. Store the raw LLM response alongside `aging_ontology.json` so you can
regenerate reconciled artefacts without another API call.

### Ontology-driven expansion (`scripts/collect_theories.py`)

**Canonical command**

```bash
python scripts/collect_theories.py "aging theory" \
    --config config/pipeline.yaml
```

**Key flags and inputs**

* `--providers` restricts retrieval to a subset of the configured sources, while
  API key overrides (e.g., `--openalex-api-key`) follow the precedence
  CLI > config > environment.
* `--parallel-fetch` and `--classification-workers` adjust concurrent provider
  fetches and LLM post-processing throughput; they fall back to sensible config
  defaults when omitted.
* Use `--no-resume` to ignore cached progress or `--state-dir` to relocate the
  persistent queue/cache (defaults to `<workdir>/collector/` when orchestrated by
  `run_pipeline.py`).
* Runtime splitting is governed by the `runtime_labels` section in
  `config/pipeline.yaml`; by default nodes with 40 or more accepted papers will
  request up to three new child and sibling proposals to keep the ontology
  balanced.

**Parallelism and staging tips**

The collector resumes automatically from its state directory—copy this folder to
checkpoint long crawls or to fan out across machines. Cached LLM responses live
under `data/cache/llm` by default; share the directory to avoid reclassifying the
same papers. Exports land where `config/pipeline.yaml` points (e.g.,
`data/pipeline/papers.csv` when launched through the orchestrators).

### Post-processing and accuracy checks

* Generate the CSV bundle and competition-friendly tables by running
  `collect_theories.py`; the helper writes `papers.csv`, `theories.csv`,
  `theory_papers.csv`, `questions.csv`, and optional competition exports defined
  in the config.
* Summarise coverage and confidence trends with:

  ```bash
  python scripts/score_progress.py --workdir data/pipeline
  ```

  The report builder loads `<workdir>/theories.csv` and `questions.csv`, then
  emits Markdown/JSON dashboards under `<workdir>/reports/`.
* Audit per-question accuracy against labelled data with:

  ```bash
  python scripts/validate_questions.py \
      --questions data/pipeline/questions.csv \
      --ground-truth path/to/answers.csv
  ```

  The tool prints a failure summary and can export a JSON report for tracking
  regressions.
* For ad-hoc literature sampling or rerunning the Q&A extractor, lean on
  `scripts/analyze_papers.py` with the same API-key override flags as the
  collector. It respects the configured caches and parallel fetchers, making it a
  convenient staging area before re-injecting results into the pipeline.

## Pipeline stages

1. **Review harvesting (`step1_*`).** PubMed, OpenAlex, and an optional Google
   Scholar collector gather review articles about aging theories. Outputs are
   merged and deduplicated into `start_reviews.json`.
2. **Filtering, full-text enrichment, and theory extraction (`step2_filter_reviews.py`).**
   Titles/abstracts are screened by an OpenAI model, retained reviews have their
   PMC/OpenAlex full texts pulled in automatically, and gpt-5-nano extracts raw
   theory mentions plus a canonical registry. This produces
   `filtered_reviews.json`, `aging_theories.json`, and (optionally)
   `<output>.failures.json` for unresolved PDFs. The legacy `step3_*`/`step4_*`
   scripts remain useful for debugging individual stages.
3. **Ontology synthesis (`step5_generate_ontology.py`).** The extracted theories
   are clustered into balanced groups/subgroups and annotated with suggested
   search keywords, producing `aging_ontology.json` plus reconciliation reports.
   Pass `--examples-per-theory <N>` to control how many representative article
   titles accompany each theory in the LLM prompt (use `0` to disable them).
4. **Ontology-driven collection (`collect_theories.py`).** Suggested queries are
   merged into the collector config, adaptive expansion discovers new keywords,
   and a global rejection registry prevents re-processing already rejected
   papers. Papers, theory assignments, and question answers are exported as CSV
   files under `<workdir>/collector/`.

All scripts can be executed individually—see their built-in `--help` text for
the full option set—allowing you to plug in new providers or debug isolated
stages without running the entire loop.

## Parallelism & performance

The pipeline is designed with six-figure corpora in mind:

* **Auto-scaling workers.** The filtering (step 2) and theory extraction (step
  4) scripts spawn one process per CPU core when the queue exceeds 100 records.
  Override this behaviour from the orchestrator with
  `--filter-processes <N>` and `--theory-processes <N>` if you need explicit
  control.
* **Chunk-aware ontology generation.** `run_pipeline.py` inspects the theory
  registry to decide how many worker processes to pass to step 5. Use
  `--ontology-processes` for manual overrides.
* **Asynchronous collectors.** `collect_theories.py` performs provider fetches,
  classification, and question extraction in parallel, enqueues new ontology
  nodes discovered during adaptive query expansion, and reuses cached LLM
  outputs on subsequent runs.
* **Global rejection registry.** Irrelevant paper identifiers are recorded once
  and skipped across future iterations, avoiding repeated LLM calls when you are
  triaging large volumes of literature.

For very large workloads, point `--state-dir` to a fast disk, keep the collector
cache (`<workdir>/collector/cache`) on SSD storage, and enable additional
providers in `config/pipeline.yaml` to diversify the corpus.

## Outputs

After the pipeline completes, the working directory contains the rich internal
artefacts plus a lightweight competition drop:

### Internal CSV exports

| File | Description |
| --- | --- |
| `papers.csv` | Full paper metadata including abstracts, joined section JSON, and full text bodies. |
| `theories.csv` | Per-theory summary with the `number_of_collected_papers` column used for progress tracking. |
| `theory_papers.csv` | Mapping between theory IDs and paper identifiers/titles. |
| `questions.csv` | Wide matrix of question answers (Q1–Q9) per theory/paper pair. |

The wide `questions.csv` export intentionally omits the legacy `<question>_confidence`
columns; confidence scores remain available in the long-form
`competition/questions.csv` table for teams that need them.

These internal tables retain every field produced by the collector and should
stay inside the workspace (they contain full texts and provenance details).

### Competition-ready tables

| File | Description |
| --- | --- |
| `competition/papers.csv` | Sanitised subset of the paper export (identifier, title, authors, abstract, source, year, DOI). |
| `competition/theories.csv` | Theory roster with a `paper_count` column matching the challenge schema. |
| `competition/theory_papers.csv` | Lightweight theory-to-paper links with paper years. |
| `competition/questions.csv` | Long-form question answers per theory/paper/question with `confidence` and optional `evidence`. |

The competition folder deliberately omits full texts and collapses questions to
a long format so that the files can be shared publicly or scored by the accuracy
module without leaking sensitive content. Sample artefacts for the Hackaging
challenge live under `data/examples/`.

## Troubleshooting

* **Missing credentials.** Each script clearly reports which environment
  variable is required when a provider key is absent.
* **OpenAI rate limits.** Use `--filter-delay`, `--theory-delay`, and the
  collector's `classification.llm.request_timeout`/`retry_backoff` settings to
  slow the request cadence.
* **Unbalanced ontology groups.** Step 5 logs every merge/split decision in the
  reconciliation report next to `aging_ontology.json`. Adjust the prompt or the
  post-processing configuration if you need tighter bounds.
* **Resuming runs.** The orchestrators skip stages whose outputs already exist.
  Delete the relevant artefacts or pass `--force` to refresh them. The collector
  keeps its state under `<workdir>/collector_state`; pass `--no-resume` in
  `run_full_cycle.py` to discard cached retrieval state.

## Repository layout

```
.
├── scripts/          # CLI entry points for each pipeline stage and orchestrators
├── src/              # Provider implementations, classifiers, ontology helpers
├── config/           # Default collector configuration
├── data/             # Example inputs and default output directory
└── tests/            # Legacy pytest suite exercising the collector components
```

Feel free to adapt the modules for additional domains—the pipeline architecture
is intentionally modular so you can extend or replace components without
rewriting the full loop.

