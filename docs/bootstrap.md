# Review bootstrap workflow

The pipeline supports an optional *bootstrap* phase that discovers new theory
labels by mining highly cited review papers before the main corpus collection
begins.  When enabled, the bootstrap step runs immediately after the pipeline
configuration and API credentials are loaded.

## Overview

1. **Query seeding** – The `corpus.bootstrap.queries` section lists one or more
   seed queries.  Each query may specify provider filters, citation thresholds,
   and cache state keys.  The pipeline renders each query using the same context
   variables that are available to theory targets (e.g. `{base_query}`).
2. **Review harvesting** – `pull_top_cited_reviews` issues the configured
   provider requests and filters the results to papers that look like reviews or
   meta-analyses.  Citation counts are extracted from the metadata (or forced via
   `citation_overrides`) and the top `max_per_query` candidates are retained per
   seed.  Normalised metadata for these papers is persisted for reproducibility.
3. **Theory extraction** – The text of each review is streamed through the
   existing `LLMClient` when available.  If no model is configured—or the API
   call fails—a deterministic heuristic scans the review for “*Theory*” headings
   and accompanying subtheory lists.  The resulting hierarchy is aggregated
   across all reviews using `build_bootstrap_ontology`.
4. **Ontology merge** – Newly discovered nodes are merged into the ontology via
   `merge_bootstrap_into_targets`.  By default the runtime retrieval targets are
   left untouched; set `corpus.bootstrap.update_targets` to `true` to make the
   new theories eligible for literature collection.
5. **Caching** – The bootstrap artefacts (queries, review metadata, and merged
   ontology) are written to `data/cache/bootstrap_ontology.json` unless a custom
   `cache_path` is supplied.  This file, together with the retrieval state,
   allows the bootstrap run to be reproduced.

## Configuration keys

Add a `bootstrap` block under `corpus` in `config/pipeline.yaml` (or your custom
configuration).  All keys are optional unless noted otherwise.

```yaml
corpus:
  bootstrap:
    enabled: true            # Defaults to true when the block is present
    queries:
      reviews:
        query: "{base_query} review"
        providers: ["openalex"]
        min_citations: 30
        max_reviews: 5
    providers: ["openalex", "crossref"]
    min_citations: 20
    limit_per_query: 40
    max_per_query: 8
    resume: true
    state_prefix: "bootstrap::reviews"
    citation_overrides:
      openalex:W12345: 120
    context:
      year: 2024
    max_theories: 6
    cache_path: data/cache/bootstrap_ontology.json
    update_targets: false
```

- **queries** *(required)*: sequence or mapping describing each seed query.
  Per-query settings override the global defaults.
- **providers**: global provider whitelist (applies when a query omits its own
  list).
- **min_citations**, **limit_per_query**, **max_per_query**, **resume**,
  **state_prefix**, **citation_overrides**, **context**, **max_theories**:
  configure the behaviour of `pull_top_cited_reviews` and
  `extract_theories_from_review`.
- **cache_path**: destination for the bootstrap cache JSON.
- **update_targets**: when `true`, merged theories are also injected into the
  runtime retrieval targets so new nodes can accumulate papers during the main
  crawl.

## Quickstart: search & auto-generate ontology

Follow the steps below to run the bootstrapper end-to-end with a single CLI
command. The example assumes you are starting from
[`config/pipeline.yaml`](../config/pipeline.yaml), which already contains a
`corpus.bootstrap` block.

1. **Provide API credentials.** Export the keys that your chosen providers
   require, or pass them directly to the collector using the CLI overrides. For
   example:

   ```bash
   export OPENALEX_API_KEY="sk-your-openalex-key"
   export CROSSREF_API_KEY="mailto:you@example.com"
   export OPENAI_API_KEY="sk-your-openai-key"  # Only needed when GPT extraction is enabled

   python scripts/collect_theories.py "geroscience" \
     --config config/pipeline.yaml \
     --quickstart \
     --target-count 60 \
     --openalex-api-key "$OPENALEX_API_KEY" \
     --crossref-api-key "$CROSSREF_API_KEY" \
     --llm-api-key "$OPENAI_API_KEY"
   ```

   The `--openalex-api-key`, `--crossref-api-key`, and `--llm-api-key` flags let
   you inject credentials without altering configuration files; omit any keys
   you exported as environment variables.

2. **Tune the quickstart node.** `--quickstart` tells the collector to create a
   transient ontology node from the CLI query, while `--target-count` sets the
   paper quota for that node. The bootstrapper augments the generated node with
   theories mined from the review search results before retrieval begins.

3. **Run the bootstrap search.** The command above renders the bootstrap queries
   defined in the configuration, fetches the matching reviews, extracts theory
   hierarchies, and merges them into both the transient quickstart node and the
   configured runtime targets. Adjust the provider list with `--providers` if
   you need to limit which APIs are hit during this exploratory pass.

The initial run writes a consolidated snapshot to
`data/cache/bootstrap_ontology.json` (or the path you set via
`corpus.bootstrap.cache_path`). The JSON payload includes the rendered queries,
review metadata, and the aggregated ontology fragment discovered during the
bootstrap step.

### Rerun enrichment or analysis from the snapshot

- Keep the generated `bootstrap_ontology.json` file under version control or in
  your working cache. Subsequent calls to `collect_theories.py` with the same
  `--state-dir` (defaults to `data/cache`) and `corpus.bootstrap.resume: true`
  reuse the cached review identifiers so providers are only queried for new
  material.
- To work purely from the cached ontology, disable the bootstrap block in your
  configuration (set `corpus.bootstrap.enabled: false`) and copy the
  `ontology` mapping from the snapshot into `corpus.targets`. You can then rerun
  `collect_theories.py` to continue filling the theory quotas or run
  `python scripts/analyze_papers.py --config config/pipeline.yaml` to refresh
  downstream analytics without hitting external APIs.
- Regenerate the snapshot at any time by re-enabling the bootstrap block and
  rerunning the quickstart command; the cache file is overwritten with the new
  hierarchy and review list.

### Troubleshooting

- **Missing API keys:** The collector raises a `MissingSecretError` if a required
  credential cannot be resolved. Double-check your environment exports and CLI
  overrides, or provide defaults under `api_keys` in the configuration file.
- **Provider rate or quota limits:** Reduce `corpus.providers[*].rate_limit_per_sec`
  or `corpus.bootstrap.limit_per_query` when you encounter HTTP 429 / 503
  responses. The `--state-dir` flag lets you persist progress between runs so
  you can continue after waiting for the provider to reset quotas.
- **No reviews discovered:** Increase `corpus.bootstrap.min_citations`, relax
  provider filters, or temporarily set `corpus.bootstrap.max_per_query` higher
  to widen the search window. Check the `reviews` block in the snapshot to see
  which filters excluded candidates.
- **LLM extraction issues:** When GPT calls fail or exceed provider limits, the
  bootstrapper falls back to deterministic pattern matching. Consider lowering
  `corpus.bootstrap.max_theories` or using a cached `llm_cache_dir` so retries
  do not duplicate costs.

For a detailed description of every parameter, consult the
[Configuration keys](#configuration-keys) section below and the inline comments
in [`config/pipeline.yaml`](../config/pipeline.yaml). Pair this guide with
[`docs/query_expansion.md`](query_expansion.md) when tuning the bootstrap module
to collaborate with the broader enrichment pipeline.

## Reproducibility tips

- Bootstrap respects the standard retrieval state directory—set
  `--state-dir` (or `corpus.cache_dir`) to reuse provider cursors.
- Override citation counts for individual papers via
  `corpus.bootstrap.citation_overrides` to eliminate ambiguity when provider
  metadata is incomplete.
- Disable bootstrap quickly by omitting the block or setting
  `corpus.bootstrap.enabled` to `false`.
