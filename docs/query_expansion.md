# Adaptive Query Expansion

The retrieval stage can optionally grow its search surface area by analysing the
papers already collected for an ontology node and synthesising new search
strings.  This behaviour is controlled by the new `QueryExpander` helper and is
fully opt-in—existing pipelines behave as before unless the configuration opts
into expansion.

## Configuration

The top-level `corpus.expansion` block defines defaults that apply to every
node.  Each theory or subtheory may override these defaults with its own
`expansion` section.  The following snippet mirrors the defaults provided in
`config/pipeline.yaml`:

```yaml
corpus:
  expansion:
    enabled: false            # Leave disabled to preserve baseline behaviour
    max_new_queries: 4        # Cap on how many adaptive shards to try per run
    max_snippets: 10          # Number of retrieved snippets to condition on
    max_gpt_queries: 3        # Upper bound on GPT-generated suggestions
    embedding_neighbors: 4    # Top embedding-derived candidates to keep
    gpt_prompt: |
      You are helping expand literature search queries for the Hackaging project.
      Return a JSON array of new search strings based on the supplied snippets.
      Avoid duplicates of the existing queries and keep each query under 12 words.
```

Per-node overrides look like this:

```yaml
corpus:
  targets:
    Activity Theory:
      expansion:
        enabled: true
        max_new_queries: 3
        max_snippets: 8
    Disengagement Theory:
      expansion:
        enabled: true
        gpt_prompt: |
          Generate at most 2 concise search strings as a JSON array.
          Focus on methodological keywords or populations that appear in the snippets.
        max_new_queries: 2
```

Set `use_gpt` or `use_embeddings` to `false` to disable a particular generator
while leaving the other enabled.  GPT prompts should return a JSON array of raw
search strings; the pipeline falls back to line-by-line parsing if the response
is not valid JSON.

## How it works

1. The collector executes the configured static queries.
2. If the node’s target remains unmet and expansion is enabled, the expander:
   - assembles snippets from the retrieved paper titles/abstracts,
   - optionally asks the configured GPT model for new search strings, and
   - mines high-value n-grams via TF‑IDF similarity as embedding neighbours.
3. The new queries are merged with the static list and re-run against the
   providers.  Additional unique papers are merged back into the state store.

Every expansion attempt is recorded as JSON under `data/cache/query_expansion/`
with the generated candidates, prompts used, snippets, and post-hoc retrieval
metrics.  This makes it easy to audit which adaptive shards improved recall and
reuse successful prompts in future runs.

## Reproducibility tips

* Commit the cache directory if you need to share the exact adaptive queries
  that were executed.
* Use descriptive prompt overrides per node so the cache log captures the
  rationale for each experimental run.
* Expansion can be disabled globally by leaving `corpus.expansion.enabled`
  unset or `false`, and re-enabled for individual nodes when exploring new
  terrain.
