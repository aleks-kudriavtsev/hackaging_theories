# Theory metadata bridge for aging biomarkers

The bridge exports the theory registry from this repository into a portable
metadata layer for `aging_biomarkers`. Theories organise search questions and
interpretation across L1 outcomes, L2 biological states, and L3 measurements.
They do not become extra L2 states and do not create causal edges by membership.

## Run

From this repository's root, use a pinned source commit:

```bash
PYTHONPATH=src python -m theories_pipeline.biomarker_bridge \
  --input data/pipeline/aging_theories.json \
  --output /tmp/theories_pack.json \
  --source-repository aleks-kudriavtsev/hackaging_theories \
  --source-commit 6e45f01e5d108d07b11ceee79de850ddfc749446
```

`aging_ontology.json` is also supported: the adapter reads `ontology.final.groups`
and representative article metadata. It does not import the raw LLM hierarchy or
the artificial quota-based splits as biological assertions. No API credentials
or new dependencies are required by the adapter itself. The existing package
initialisation and its dependencies still apply when using module invocation.

## Contract and source handling

The output is schema `1.0`, layer `theory_metadata`, with `theories`, empty
`empirical_edges` and `theory_links`, and provenance recording the source repository,
full Git commit, input file and SHA-256 of its actual bytes. The supplied commit
is a provenance assertion by the caller; it is not independently checked against
Git by this offline adapter. Use a clean checkout at the declared commit.

Each theory preserves its source ID, label and aliases without merging concepts.
`theory_class` is copied only when explicitly provided as `evolutionary_theory`,
`mechanistic_hypothesis`, `conceptual_framework`, `intervention`, or `unknown`.
Labels and literature quantity do not determine class or scientific validity.

Nodes have one of two import statuses:

| Status | Meaning |
| --- | --- |
| `hypothesis_candidate` | Candidate concept; source identifiers still require bibliographic and claim-level review. |
| `quarantined_fixture` | Example/synthetic provenance or suspect reference; excluded from scientific evidence and panel selection. |

The checked-in `aging_theories.json` and `aging_ontology.json` contain three
demonstration references with DOI values `10.1000/aging.001`–`003`. They are all
quarantined. `A1`-style references without a DOI or explicit PMID, fixture/example
input paths, and explicit synthetic metadata are also quarantined. This is a
conservative import filter, not an exhaustive fabrication detector. A
non-placeholder DOI can still be nonexistent or irrelevant.

Citation identifiers are never resolved or invented here. A DOI is not converted
into a PMID. An explicit PMID can be copied, but it remains `identifier_unverified`.
LLM confidence, keyword agreement and upstream `validated` flags do not change
that status. Duplicate IDs with conflicting content fail instead of overwriting.

To correct a quarantined record, provide a new source snapshot with the actual
reference and export it again. The receiving system keeps the old provenance and
an explicit review trail; changing a source does not silently validate a theory.

## Empirical connection contract

The receiving evidence system should store separate, reviewable records:

| Entity | Required scientific content |
| --- | --- |
| Theory | Definition, scope, competing explanations and falsifiable predictions. |
| Claim | Subject, predicate, object, direction, population/species, tissue, study design, endpoint and uncertainty. |
| Source | Resolved DOI/PMID, retrieval date, source hash, correction/retraction status and exact evidence location. |
| Theory link | Theory ID, claim ID, relation (`predicts`, `supports`, `contradicts`, `qualifies`), rationale and review status. |
| Panel proposal | Candidate measurands and coverage, replication, independence/redundancy, assay feasibility and intended-use constraints. |

A theory may link to all three empirical levels through claims. A protein found
in a paper about a theory is only a search candidate until the actual measurement,
outcome association and analytical feasibility have been assessed. Theory coverage
can guide diversity of a research panel; it cannot establish clinical utility.

Keep `supports` distinct from `mentions`; observations distinct from interventions;
human endpoints distinct from lifespan results in model organisms. Retrieve
contradictory and null findings as well as supportive ones. Deduplicate cohorts,
not just papers, before treating findings as independent replications.

## Reusable upstream mechanisms and boundaries

- Reuse `QueryExpansionCache`, `QueryExpander.expand/record_performance` for
  traceable search proposals and retrieval yield. Yield is not evidence strength.
- Reuse `OntologyManager` change notifications for candidate discovery. Do not
  auto-promote generated concepts to accepted science.
- Reuse `question_validation.validate` with a separately curated evaluation set.
  The small bundled fixture is a software test, not a scientific validation set.
- Do not use `OntologyOptimizer._fallback_decision` to construct biological
  theories: it creates quota-sized `Variant` chunks on LLM failure.
- Do not use `QuestionExtractor` confidence as an evidence grade. The Q3 keyword
  fallback has been corrected: intervention mentions, including positive, null
  and negative findings, now use the legacy `Proposed longevity intervention`
  category with `screening_only=true`, `efficacy_assessed=false` and the reason
  `requires source-level review` in heuristic provenance. Verbatim sentences and
  their negation are preserved. Outcome words or numerical effects never produce
  `Validated longevity intervention` through this fallback. The LLM branch and
  the other question heuristics have not been scientifically validated; the LLM
  can still return the old validation label, so a separate claim review remains
  mandatory. Heuristic/LLM agreement is an uncalibrated confidence score.

Run the focused boundary checks with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_biomarker_bridge.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_intervention_claim_guards.py -v
```

## Curated literature foundation (2026-09-06)

`data/curated/aging_theory_catalog.json` now provides a real, separately curated
discovery input. It does not replace or promote the old demo registry. Selection
is purposeful rather than a systematic or exhaustive inventory of aging theories.
No project-report panel was used as an empirical evidence source.

| Concept | Type | Foundation and scope |
| --- | --- | --- |
| Inflammaging | Conceptual framework | Franceschi et al., 2000; original concept, PMID 10911963. |
| Antagonistic pleiotropy | Evolutionary theory | Williams, 1957; DOI 10.1111/j.1558-5646.1957.tb02911.x. |
| Hyperfunction | Mechanistic hypothesis | Blagosklonny, 2006; original hypothesis, PMID 17012837. |
| Free-radical damage | Mechanistic hypothesis | Harman, 1956, PMID 13332224; a context-specific counterexample, Van Raamsdonk and Hekimi, 2009, PMID 19197346, is retained. |
| Telomere attrition | Mechanistic hypothesis | Harley et al., 1990, PMID 2342578; cultured human fibroblasts, not a validated organismal clock. |
| Calorie restriction | Intervention family | Historical rat experiment reprinted in 1989, PMID 2520283; human metabolic intervention, Redman et al., 2018, PMID 29576535. |

The eight source records include original conceptual publications and primary
experiments. Seven PubMed records were retrieved together from Europe PMC;
the response bytes, request, expected IDs and SHA-256 are retained in
`data/curated/bibliography/`. Williams' identity was checked on the journal
record. Bibliographic identity and conceptual relevance were reviewed; a
systematic correction/retraction review and clinical-claim review were not
performed by this upstream catalog.

The consumer configuration contains six concepts, 18 context nodes, 24 explicit
hypothesis links, seven falsifiable research questions and six paired search
plans. Every concept has L3 measurement, L2 state and L1 outcome context. The
four protein candidates TNF, IL6, CRP and GDF15 are accompanied by four proposed
additional measurements: leukocyte telomere length, cellular phospho-S6,
urinary F2-isoprostanes and leptin. These additions are marked `outside_panel`;
they do not enter a protein panel automatically. GDF15 is explicitly a
nonspecific stress comparator, not a direct readout of mTOR or oxidative damage.

Each question specifies its competing explanation, required discriminating
test and falsification criterion. All upstream links remain `organizes` or
`proposes_marker`, with empty observation references. No empirical edge,
confirmed prediction, clinical priority or evidence score is exported.

The six existing theory-query term lists and original prediction-query term
sets are preserved. The receiver's six theory queries and shared contradiction
query are byte-for-byte equivalent to the existing baseline. Additional paired
search plans are proposals and are not reported as completed searches.

### Export and integrity contract

From a committed checkout, use that commit's complete hash:

```bash
PYTHONPATH=src python -m theories_pipeline.biomarker_bridge \
  --input data/curated/aging_theory_catalog.json \
  --output /tmp/curated_theories_pack.json \
  --consumer-output /tmp/theory_config.json \
  --source-repository aleks-kudriavtsev/hackaging_theories \
  --source-commit FULL_COMMIT_HASH
```

Legacy schema `1.0` fields and conservative citation statuses are unchanged.
Curated packs additionally contain `consumer_config`, directly accepted by
`aging_biomarkers/tools/living_evidence/theories.py`, and `discovery_summary`.
Bibliographic review records are separate from theory-validation status.

`pack.provenance` contains `source_repository`, `source_commit`, `input_file`,
`registry_sha256` and `consumer_config_sha256`. The last hash is SHA-256 of
UTF-8 JSON with `ensure_ascii=False`, `sort_keys=True`, and
`separators=(",", ":")`. The catalog pins the complete `consumer_config` object.
The exporter independently rebuilds and validates it and requires exact
equality. A receiver can independently hash `git show COMMIT:input_file` and
compare its `consumer_config` to the pack. This binds discovery configuration
to source bytes, rather than trusting a self-declared hash alone.

After editing curated source fields, regenerate the derived pinned object and
review the diff before committing:

```bash
PYTHONPATH=src python scripts/refresh_curated_consumer.py \
  data/curated/aging_theory_catalog.json
PYTHONPATH=src python -m unittest discover -s tests -p test_curated_catalog.py -v
```

Malformed references, unreviewed identities, unknown graph nodes/analytes,
empirical assertions, confirmed predictions and drift of the derived config
fail closed. The receiver still owns literature retrieval, claim review,
cohort deduplication, analytical feasibility and three-level panel selection.
