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

### API и форматы учётных данных

| API / сервис | Назначение | Переменная/флаг | Формат учётных данных |
| --- | --- | --- | --- |
| OpenAlex | Базовый поиск статей и обзоров | `OPENALEX_API_KEY`, `--openalex-api-key` | Строка API-токена OpenAlex |【F:config/pipeline.yaml†L1-L37】【F:scripts/collect_theories.py†L838-L841】
| Crossref | Дополнительные метаданные и цитирования | `CROSSREF_API_KEY`, `--crossref-api-key` | Контактный email в формате `mailto:you@example.com` |【F:config/pipeline.yaml†L3-L31】【F:scripts/collect_theories.py†L841-L844】
| PubMed | Биомедицинские обзоры и статьи | `PUBMED_API_KEY`, `--pubmed-api-key` | API-ключ NCBI (ASCII-строка) |【F:config/pipeline.yaml†L5-L37】【F:scripts/collect_theories.py†L844-L847】
| Sci-Hub (RapidAPI) | Поиск полнотекстовых зеркал DOI | `SCIHUB_RAPIDAPI_KEY`, `--scihub-rapidapi-key` | RapidAPI token (`X-RapidAPI-Key`) |【F:config/pipeline.yaml†L11-L48】【F:src/theories_pipeline/literature.py†L684-L779】【F:scripts/collect_theories.py†L847-L852】
| Sci-Hub (библиотечный клиент) | Альтернатива RapidAPI | `SCIHUB_EMAIL`, `--scihub-email` | Email-адрес для `scihub.py` |【F:config/pipeline.yaml†L9-L48】【F:src/theories_pipeline/literature.py†L684-L779】【F:scripts/collect_theories.py†L844-L850】
| Anna’s Archive | Альтернативные зеркала PDF | `ANNAS_ARCHIVE_API_KEY`, `--annas-archive-api-key` | RapidAPI token (`X-RapidAPI-Key`) |【F:config/pipeline.yaml†L12-L68】【F:src/theories_pipeline/literature.py†L841-L937】【F:scripts/collect_theories.py†L850-L853】
| OpenAI (GPT) | Классификация теорий и ответы Q1–Q9 | `OPENAI_API_KEY`, `--llm-api-key`, `--llm-model` | API-ключ OpenAI + имя модели (например, `gpt-4o-mini`) |【F:config/pipeline.yaml†L69-L142】【F:scripts/collect_theories.py†L786-L871】

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

### Quickstart without a seed ontology

You can explore the pipeline without curating an ontology file up front. Export
any required provider credentials (or pass them inline with the CLI overrides)
and invoke the collector in quickstart mode:

```bash
export OPENALEX_API_KEY="sk-your-openalex-key"
export CROSSREF_API_KEY="mailto:you@example.com"

python scripts/collect_theories.py "Aging Theory" \
  --quickstart \
  --target-count 75 \
  --openalex-api-key "$OPENALEX_API_KEY" \
  --crossref-api-key "$CROSSREF_API_KEY"
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
for advanced enrichment strategies.

### Quickstart: автоматическая генерация онтологии

Follow the steps below to let the bootstrapper discover theories automatically and
attach them to a transient ontology node generated from your CLI query.

1. **Подготовьте переменные окружения и флаги.** Укажите ключи для всех задействованных
   провайдеров через переменные окружения или одноимённые CLI-флаги (`--openalex-api-key`,
   `--crossref-api-key`, `--pubmed-api-key`, `--scihub-rapidapi-key`, `--scihub-email`,
   `--annas-archive-api-key`). Эти ключи сопоставляются с записями в блоке `api_keys`
   конфигурации, поэтому подходят как токены RapidAPI, так и контактный адрес для
   Crossref (формат `mailto:you@example.com`). Укажите также `OPENAI_API_KEY` и модель
   через `--llm-model`, если хотите задействовать GPT для классификации и извлечения.
   【F:config/pipeline.yaml†L1-L83】【F:scripts/collect_theories.py†L786-L871】【F:scripts/collect_theories.py†L904-L951】
2. **Включите нужные провайдеры.** Запрос можно ограничить конкретными источниками при
   помощи `--providers openalex crossref scihub annas_archive`, либо оставить список по
   умолчанию из конфигурации. Полнотекстовые зеркала (Sci-Hub, Anna’s Archive) и PubMed
   отключены по умолчанию; активируйте их в `config/pipeline.yaml` или через CLI, чтобы
   bootstrap получил доступ к PDF и обзорам. 【F:config/pipeline.yaml†L19-L84】【F:scripts/collect_theories.py†L829-L833】【F:src/theories_pipeline/literature.py†L660-L1030】
3. **Запустите bootstrap + сбор.** Быстрый старт генерирует временной узел онтологии,
   выполняет bootstrap-поиск обзоров, извлекает из них теории при помощи LLM (с падением
   обратно на детерминированные эвристики, если модель недоступна), а затем переходит к
   основной выборке литературы. Команда ниже включает кэширование и ограничивает
   провайдеры RapidAPI, чтобы быстрее наполнить узел:

   ```bash
   export OPENALEX_API_KEY="sk-your-openalex-key"
   export CROSSREF_API_KEY="mailto:you@example.com"
   export OPENAI_API_KEY="sk-your-openai-key"
   export SCIHUB_RAPIDAPI_KEY="your-rapidapi-token"
   export ANNAS_ARCHIVE_API_KEY="your-rapidapi-token"

   python scripts/collect_theories.py "geroscience" \
     --config config/pipeline.yaml \
     --quickstart \
     --target-count 60 \
     --providers openalex crossref scihub annas_archive \
     --llm-model gpt-4o-mini \
     --llm-api-key "$OPENAI_API_KEY"
   ```

   Снэпшот онтологии и метаданные bootstrap сохраняются в `data/cache/bootstrap_ontology.json`,
   а собранные статьи — в каталоге, указанном в блоке `outputs`. Повторный запуск с тем же
   `--state-dir` продолжит наполнение узла, используя кэш. 【F:docs/bootstrap.md†L1-L119】【F:scripts/collect_theories.py†L720-L823】【F:scripts/collect_theories.py†L951-L1020】

### Пример запуска: bootstrap → сбор → анализ

```bash
# 1. Bootstrap и генерация временного узла онтологии из запроса
python scripts/collect_theories.py "geroscience" --config config/pipeline.yaml --quickstart --target-count 60

# 2. Повторный сбор по обновлённой онтологии (можно включить дополнительные провайдеры)
python scripts/collect_theories.py "activity engagement" --config config/pipeline.yaml --providers openalex crossref pubmed

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
