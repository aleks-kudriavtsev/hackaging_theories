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

The project only depends on the standard library plus a small set of helper
packages used for configuration parsing and tests:

```bash
pip install pyyaml pytest
```

Install any additional providers (e.g., HTTP clients) required for your custom
retrieval strategies in the same environment.

Optional helpers:

- `pdfminer.six` — enables PDF-to-text extraction when providers only expose
  binary downloads for full texts.

### Environment variables

External API keys are declared in the configuration under `api_keys`, and the
CLI entry points resolve them lazily via environment variables or external
files. Export the following variables before running the pipeline to keep
secrets out of version control:

- `OPENALEX_API_KEY`
- `CROSSREF_API_KEY`
- `PUBMED_API_KEY`
- `OPENAI_API_KEY`
- `SCIHUB_EMAIL` (used when delegating lookups to the optional `scihub.py` client)
- `SCIHUB_RAPIDAPI_KEY`
- `ANNAS_ARCHIVE_API_KEY`

Each key supports fallbacks (defaults or file references) if you need a more
specialised workflow. See `src/theories_pipeline/config_utils.py` for the
supported descriptors and `config/pipeline.yaml` for annotated examples.

### Credential reference tables

Use the tables below to map each supported provider to its credential source and
the configuration or CLI hook that consumes it. All variables live under the
`api_keys` block in [`config/pipeline.yaml`](config/pipeline.yaml) and can be
overridden via the collector flags shown in the last column.

#### Scholarly metadata and search APIs

| Provider | Credential & where to request it legitimately | Configuration reference | CLI override |
| --- | --- | --- | --- |
| OpenAlex | Free account at [https://docs.openalex.org/](https://docs.openalex.org/) to generate a personal access token. | `api_keys.openalex.env: OPENALEX_API_KEY`; provider `openalex.api_key_key: openalex`. 【F:config/pipeline.yaml†L1-L29】 | `--openalex-api-key` |【F:scripts/collect_theories.py†L1080-L1091】
| Crossref | Register a mailto-style contact per the [Crossref polite usage policy](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-metadata-service-guidelines/). | `api_keys.crossref_contact.env: CROSSREF_API_KEY`; provider `crossref.api_key_key: crossref_contact`. 【F:config/pipeline.yaml†L3-L35】 | `--crossref-api-key` |【F:scripts/collect_theories.py†L1091-L1095】
| PubMed (NCBI E-utilities) | Request an API key via the NCBI account dashboard. | `api_keys.pubmed.env: PUBMED_API_KEY`; provider `pubmed.api_key_key: pubmed`. 【F:config/pipeline.yaml†L5-L61】 | `--pubmed-api-key` |【F:scripts/collect_theories.py†L1095-L1099】
| SerpApi (Google Scholar bridge) | Subscribe at [https://serpapi.com/](https://serpapi.com/) for a private API key. | `api_keys.serpapi.env: SERPAPI_KEY`; provider `serpapi_scholar.api_key_key: serpapi`. 【F:config/pipeline.yaml†L7-L49】 | `--serpapi-key` |【F:scripts/collect_theories.py†L1099-L1103】
| Semantic Scholar | Apply for a research API key through the Semantic Scholar portal. | `api_keys.semantic_scholar.env: SEMANTIC_SCHOLAR_KEY`; provider `semantic_scholar.api_key_key: semantic_scholar`. 【F:config/pipeline.yaml†L9-L57】 | `--semantic-scholar-key` |【F:scripts/collect_theories.py†L1103-L1107】

OpenAlex tokens are appended to requests as an `api_key` query parameter (rather than a bearer header) alongside an optional `mailto` contact when configured. 【F:src/theories_pipeline/literature.py†L430-L436】

#### Full-text resolvers and mirrors

| Provider | Credential & where to request it legitimately | Configuration reference | CLI override |
| --- | --- | --- | --- |
| Sci-Hub email (community client) | Supply the contact address permitted by your institutional access policy. | `api_keys.scihub_email.env: SCIHUB_EMAIL`; provider `scihub.extra.email_key: scihub_email`. 【F:config/pipeline.yaml†L11-L66】 | `--scihub-email` |【F:scripts/collect_theories.py†L1107-L1111】
| Sci-Hub RapidAPI | Purchase a subscription through [https://rapidapi.com/](https://rapidapi.com/). | `api_keys.scihub_rapidapi.env: SCIHUB_RAPIDAPI_KEY`; provider `scihub.api_key_key: scihub_rapidapi`. 【F:config/pipeline.yaml†L11-L66】 | `--scihub-rapidapi-key` |【F:scripts/collect_theories.py†L1111-L1115】
| Anna's Archive | Subscribe to the Anna's Archive RapidAPI product for a token. | `api_keys.annas_archive.env: ANNAS_ARCHIVE_API_KEY`; provider `annas_archive.api_key_key: annas_archive`. 【F:config/pipeline.yaml†L12-L68】 | `--annas-archive-api-key` |【F:scripts/collect_theories.py†L1115-L1119】

#### Language models

| Provider | Credential & where to request it legitimately | Configuration reference | CLI override |
| --- | --- | --- | --- |
| OpenAI | Create an API key via the [OpenAI dashboard](https://platform.openai.com/account/api-keys). | `api_keys.openai.env: OPENAI_API_KEY`; `classification.llm.model` and `api_key_key: openai`. 【F:config/pipeline.yaml†L13-L92】 | `--llm-api-key`, `--llm-model` |【F:scripts/collect_theories.py†L1119-L1149】

### Советы по устранению ошибок

- **Отсутствующие ключи.** Скрипт завершается с `MissingSecretError`, если не удаётся сопоставить требуемый ключ из конфигурации с переменной окружения или аргументом. Проверьте экспорт и блок `api_keys`. 【F:scripts/collect_theories.py†L899-L915】【F:docs/bootstrap.md†L120-L139】
- **Лимиты провайдеров (HTTP 429/503).** Снизьте `rate_limit_per_sec` для источников в конфигурации или ограничьте список провайдеров через `--providers`, затем перезапустите сбор после паузы. `--state-dir` сохраняет прогресс, чтобы не терять уже собранные результаты. 【F:config/pipeline.yaml†L19-L83】【F:docs/bootstrap.md†L132-L150】
- **Не найдены обзоры для bootstrap.** Ослабьте фильтры (`min_citations`, `max_per_query`) или расширьте провайдеры, чтобы увеличить окно поиска. Проверяйте `data/cache/bootstrap_ontology.json` для диагностики. 【F:docs/bootstrap.md†L100-L150】
- **Ошибки LLM или превышение квот.** Установите кэш `llm.cache_dir` / `extraction.llm.cache_dir` и уменьшите размер батчей через `--llm-batch-size`, чтобы сократить повторные обращения. При недоступности модели пайплайн автоматически переходит к эвристикам. 【F:config/pipeline.yaml†L69-L142】【F:docs/bootstrap.md†L100-L150】

### Preprint providers, rate limits, and full-text mirrors

The pipeline ships with optional bioRxiv and medRxiv providers that fetch
preprint metadata via the public JSON endpoints. These sources are disabled by
default; enable them in `config/pipeline.yaml` once you have configured a
conservative rate limit (≤ 1 request/second is recommended by the operators)
and a manageable date window. Category filters can further narrow the feed so
the retriever only inspects relevant domains. Refer to the inline comments in
`config/pipeline.yaml` for example settings.

Two additional providers help resolve full texts when primary APIs only supply
metadata. The Sci-Hub integration works with either RapidAPI or the community
`scihub.py` client—configure the relevant credentials via
`SCIHUB_RAPIDAPI_KEY`/`SCIHUB_EMAIL` and tune request headers by editing the
`providers[].extra` section. Anna's Archive is exposed through RapidAPI and
accepts custom link keys for grabbing mirror URLs. Both providers respect the
shared full-text cache under `data/cache/fulltext` and should be rate limited
to avoid overwhelming upstream mirrors.

## Running the pipelines

### Quickstart: Generate an ontology from reviews

1. **Set provider credentials.** Export environment variables (or prepare to
   supply the CLI overrides) for every API you intend to call during the
   bootstrap run. The quickstart workflow currently relies on PubMed search to
   find review articles, so make sure those credentials are available before
   running:
   - `OPENALEX_API_KEY` / `--openalex-api-key`
   - `PUBMED_API_KEY` / `--pubmed-api-key`
   - `SERPAPI_KEY` / `--serpapi-key`
   - `SEMANTIC_SCHOLAR_KEY` / `--semantic-scholar-key`
   - `SCIHUB_EMAIL` / `--scihub-email`
   - `SCIHUB_RAPIDAPI_KEY` / `--scihub-rapidapi-key`
   - `ANNAS_ARCHIVE_API_KEY` / `--annas-archive-api-key`
    - `OPENAI_API_KEY` / `--llm-api-key` (plus `--llm-model` when you opt into GPT
      classification)

Crossref remains supported as an optional metadata source—include it only when
necessary by adding `--providers crossref` along with the corresponding
credential from the provider table if your run requires that metadata.
2. **Run the bootstrapper and enrichment pipeline.** The command below assumes
   the default configuration file already contains a `corpus.bootstrap` block
   and writes the ontology snapshot to
   `data/cache/bootstrap_ontology.json`:

   ```bash
   export OPENALEX_API_KEY="sk-your-openalex-key"
   export PUBMED_API_KEY="your-pubmed-key"
   export OPENAI_API_KEY="sk-your-openai-key"

   python scripts/collect_theories.py "geroscience" \
     --config config/pipeline.yaml \
     --quickstart \
     --target-count 60 \
     --openalex-api-key "$OPENALEX_API_KEY" \
     --pubmed-api-key "$PUBMED_API_KEY" \
     --llm-api-key "$OPENAI_API_KEY"
   ```

   Add the other overrides from the list above whenever you enable the
   associated providers in your configuration. Include Crossref only if needed
   by extending `--providers` and supplying its contact credential. 【F:docs/bootstrap.md†L78-L114】

### Quickstart without a seed ontology

You can explore the pipeline without curating an ontology file up front. Export
any required provider credentials (or pass them inline with the CLI overrides)
and invoke the collector in quickstart mode. As above, PubMed powers the
default review discovery:

```bash
export OPENALEX_API_KEY="sk-your-openalex-key"
export PUBMED_API_KEY="your-pubmed-key"

python scripts/collect_theories.py "Aging Theory" \
  --quickstart \
  --target-count 75 \
  --openalex-api-key "$OPENALEX_API_KEY" \
  --pubmed-api-key "$PUBMED_API_KEY"
```

This command renders an ad-hoc ontology node named after the query, writes the
definition to `data/cache/ontologies/aging-theory.json`, and then resumes the
standard retrieval → classification → extraction flow. Outputs are emitted to
the paths configured under `outputs` in `config/pipeline.yaml`
(`data/examples/papers.csv`, `data/examples/theories.csv`, and
`data/examples/questions.csv` by default). Subsequent enrichment runs reuse the
same cache directories (`data/cache/literature` for provider state and
`data/cache/ontologies/` for the generated node), so re-running the command will
continue filling the target quota unless `--no-resume` is supplied. Promote the
saved ontology JSON into `corpus.targets` in your configuration when you are
ready to graduate to a managed ontology; see
[`config/pipeline.yaml`](config/pipeline.yaml) for the fully managed layout and
[`docs/bootstrap.md`](docs/bootstrap.md) / [`docs/query_expansion.md`](docs/query_expansion.md)
for advanced enrichment strategies. Add Crossref to the provider list only when
you require its metadata for follow-up runs.

### Quickstart: автоматическая генерация онтологии

Follow the steps below to let the bootstrapper discover theories automatically and
attach them to a transient ontology node generated from your CLI query.

1. **Подготовьте переменные окружения и флаги.** Укажите ключи для всех задействованных
   провайдеров через переменные окружения или одноимённые CLI-флаги (`--openalex-api-key`,
   `--pubmed-api-key`, `--scihub-rapidapi-key`, `--scihub-email`, `--annas-archive-api-key`).
   Быстрый старт использует PubMed для поиска обзорных статей, поэтому убедитесь, что
   ключ `PUBMED_API_KEY` доступен. Эти ключи сопоставляются с записями в блоке `api_keys`
   конфигурации. Укажите также `OPENAI_API_KEY` и модель через `--llm-model`, если хотите
   задействовать GPT для классификации и извлечения.
   【F:config/pipeline.yaml†L1-L83】【F:scripts/collect_theories.py†L786-L871】【F:scripts/collect_theories.py†L904-L951】
2. **Включите нужные провайдеры.** Запрос можно ограничить конкретными источниками при
   помощи `--providers openalex pubmed scihub annas_archive`, либо оставить список по
   умолчанию из конфигурации. Полнотекстовые зеркала (Sci-Hub, Anna’s Archive) и PubMed
   отключены по умолчанию; активируйте их в `config/pipeline.yaml` или через CLI, чтобы
   bootstrap получил доступ к PDF и обзорам. Crossref можно подключить дополнительно,
   если вам нужна эта метадата. 【F:config/pipeline.yaml†L19-L84】【F:scripts/collect_theories.py†L829-L833】【F:src/theories_pipeline/literature.py†L660-L1030】
3. **Запустите bootstrap + сбор.** Быстрый старт генерирует временной узел онтологии,
   выполняет bootstrap-поиск обзоров, извлекает из них теории при помощи LLM (с падением
   обратно на детерминированные эвристики, если модель недоступна), а затем переходит к
   основной выборке литературы. Команда ниже включает кэширование и ограничивает
   провайдеры RapidAPI, чтобы быстрее наполнить узел:

   ```bash
   export OPENALEX_API_KEY="sk-your-openalex-key"
   export PUBMED_API_KEY="your-pubmed-key"
   export OPENAI_API_KEY="sk-your-openai-key"
   export SCIHUB_RAPIDAPI_KEY="your-rapidapi-token"
   export ANNAS_ARCHIVE_API_KEY="your-rapidapi-token"

   python scripts/collect_theories.py "geroscience" \
     --config config/pipeline.yaml \
     --quickstart \
     --target-count 60 \
     --providers openalex pubmed scihub annas_archive \
     --llm-model gpt-4o-mini \
     --llm-api-key "$OPENAI_API_KEY"
   ```

   Снэпшот онтологии и метаданные bootstrap сохраняются в `data/cache/bootstrap_ontology.json`,
   а собранные статьи — в каталоге, указанном в блоке `outputs`. Повторный запуск с тем же
   `--state-dir` продолжит наполнение узла, используя кэш. 【F:docs/bootstrap.md†L1-L119】【F:scripts/collect_theories.py†L720-L823】【F:scripts/collect_theories.py†L951-L1020】

### Work with the cached bootstrap ontology

- **Inspect.** Pretty-print the snapshot with `python -m json.tool data/cache/bootstrap_ontology.json`
  or `jq '.' data/cache/bootstrap_ontology.json` to review the generated queries,
  review metadata, and ontology fragment. 【F:docs/bootstrap.md†L116-L120】
- **Amend.** Edit the `ontology` block directly (for example, to tweak theory
  labels or citation counts) and copy the adjusted mapping into
  `corpus.targets` when you are ready to manage the hierarchy manually. 【F:docs/bootstrap.md†L129-L134】
- **Reuse.** Keep the cache under version control or a shared working directory
  so subsequent runs with the same `--state-dir` and `corpus.bootstrap.resume:
  true` reuse review identifiers instead of hitting provider APIs again. Disable
  the bootstrap block (`corpus.bootstrap.enabled: false`) to run enrichment
  purely from the cached ontology, or re-enable it to refresh the snapshot on
  demand. 【F:docs/bootstrap.md†L124-L137】【F:scripts/collect_theories.py†L1230-L1276】

### Пример запуска: bootstrap → сбор → анализ

```bash
# 1. Bootstrap и генерация временного узла онтологии из запроса
python scripts/collect_theories.py "geroscience" --config config/pipeline.yaml --quickstart --target-count 60

# 2. Повторный сбор по обновлённой онтологии (можно включить дополнительные провайдеры)
python scripts/collect_theories.py "activity engagement" --config config/pipeline.yaml --providers openalex pubmed

# 3. Аналитика и обновление сводных отчётов
python scripts/analyze_papers.py --config config/pipeline.yaml
```

Используйте `--state-dir data/cache` для совместного кэша между командами, чтобы не
переизвлекать уже найденные записи. 【F:scripts/collect_theories.py†L720-L823】【F:scripts/collect_theories.py†L904-L1020】【F:scripts/analyze_papers.py†L161-L196】

### Collect theories and initial Q1–Q9 answers

```bash
python scripts/collect_theories.py "activity engagement" --config config/pipeline.yaml
```

This command

1. Loads seed papers from `config/pipeline.yaml` via
   `LiteratureRetriever.search()`.
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
