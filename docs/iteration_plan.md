# Iterative Coverage Growth Plan

The Hackaging literature pipeline should be rerun on a regular cadence to keep the
ontology and downstream analyses current. The cycle below aligns with the updated
configuration in `config/pipeline.yaml` and emphasises checkpoints for cache health
and duplicate control.

## Step 1 – Provider activation & credential refresh
- Confirm the API keys listed under `api_keys` resolve (`openalex`, `pubmed`,
  `semantic_scholar`, and `serpapi`). Override secrets via environment variables or
  CLI flags before each run to avoid expired credentials.
- Review provider quotas and adjust per-provider `rate_limit_per_sec` or `batch_size`
  if any service is throttling.

## Step 2 – Bootstrap query sweep
- Update the seed templates inside `corpus.bootstrap.queries` when new theory angles
  emerge; focus on high-citation reviews of aging mechanisms.
- Run `python scripts/run_full_cycle.py --workdir data/pipeline --force` to rebuild
  `data/cache/bootstrap_ontology.json`. Inspect the `queries` and `reviews` sections in
  the cache file to confirm the new seeds were executed and captured.

## Step 3 – Filtering & classification pass
- After bootstrap completes, allow the classifier to filter noise by checking
  `data/pipeline/cache/theories.json` (or the configured cache dir) for LLM decisions.
- Adjust `classification.llm` settings or keywords when the acceptance rate dips below
  expectations.

## Step 4 – Full-text retrieval & ontology enrichment
- Monitor the collector output in `<workdir>/papers.csv` to ensure full-text URLs are
  resolving. Re-run specific providers with `--providers` filters when gaps appear.
- Verify the refreshed ontology snapshot at `<workdir>/aging_ontology.json` captures
  new nodes from the bootstrap stage before proceeding to expansion.

## Step 5 – Adaptive query expansion review
- Keep `corpus.expansion.enabled` set to `true` and audit each generated batch inside
  `data/cache/query_expansion/`. Remove or blacklist query variants that repeatedly
  return duplicates or off-topic hits by editing the node-level `expansion` blocks.
- Document successful prompts and snippet parameters inside the cache folder so they
  can be reused or promoted to static queries.

## Cadence & governance
- Target a full refresh every 4–6 weeks, with interim expansion-only runs when new
  literature surges are expected (e.g. post-conference releases).
- Track run metadata (command used, keys enabled, notable configuration tweaks) in a
  shared changelog so subsequent cycles can reuse the most effective settings.
