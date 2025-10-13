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

## Reproducibility tips

- Bootstrap respects the standard retrieval state directory—set
  `--state-dir` (or `corpus.cache_dir`) to reuse provider cursors.
- Override citation counts for individual papers via
  `corpus.bootstrap.citation_overrides` to eliminate ambiguity when provider
  metadata is incomplete.
- Disable bootstrap quickly by omitting the block or setting
  `corpus.bootstrap.enabled` to `false`.
