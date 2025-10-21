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

## Pipeline stages

1. **Review harvesting (`step1_*`).** PubMed, OpenAlex, and an optional Google
   Scholar collector gather review articles about aging theories. Outputs are
   merged and deduplicated into `start_reviews.json`.
2. **Relevance filtering (`step2_filter_reviews.py`).** Titles/abstracts are
   screened by an OpenAI model, with batched requests and multi-process workers
   for large queues. Decisions are stored in `filtered_reviews.json`.
3. **Full-text enrichment (`step3_fetch_fulltext.py`).** PubMed Central (PMC)
   open-access copies are downloaded when available and normalised to plain
   text, saving results to `filtered_reviews_fulltext.json`. The script batches
   PMCID lookups by default—200 articles per Entrez `efetch` call—to stay within
   NCBI response limits while keeping 10k–100k review queues manageable. Override
   this with `--entrez-batch-size <N>` or `PUBMED_BATCH_SIZE=<N>` if you need a
   different cadence.
4. **Theory extraction (`step4_extract_theories.py`).** Long reviews are chunked
   and sent to the LLM to identify aging theories. Outputs include per-review
   annotations and a consolidated registry in `aging_theories.json`.
5. **Ontology synthesis (`step5_generate_ontology.py`).** The extracted theories
   are clustered into balanced groups/subgroups and annotated with suggested
   search keywords, producing `aging_ontology.json` plus reconciliation reports.
   Pass `--examples-per-theory <N>` to control how many representative article
   titles accompany each theory in the LLM prompt (use `0` to disable them).
6. **Ontology-driven collection (`collect_theories.py`).** Suggested queries are
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

After the pipeline completes, the working directory contains:

| File | Description |
| --- | --- |
| `start_reviews*.json` | Raw review metadata from each provider. |
| `filtered_reviews.json` | LLM-filtered review subset with explanations. |
| `filtered_reviews_fulltext.json` | Reviews augmented with PMC full texts. |
| `aging_theories.json` | Theory annotations and the consolidated registry. |
| `aging_ontology.json` | Balanced ontology with suggested queries and metrics. |
| `collector/` | Papers, theory assignments, Q&A CSVs, cache, and rejection registry. |

Sample artefacts for the Hackaging challenge live under `data/examples/`.

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

