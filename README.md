# Hackaging Theories Pipeline

This repository contains a reproducible pipeline for the Hackaging challenge. It
includes modular tooling for literature retrieval, theory classification, the
standard nine-question extraction workflow, and CSV export utilities.

## Project layout

```
.
├── config/                 # YAML/JSON configuration for reproducible runs
├── data/examples/          # Seed dataset and sample CSV outputs
├── scripts/                # Command line entry points for the pipeline
├── src/theories_pipeline/  # Core Python package with reusable modules
└── tests/                  # Automated pytest suite with mocked inputs
```

## Getting started

1. Ensure Python 3.10+ is available.
2. Install dependencies if you plan to parse the YAML config:

   ```bash
   pip install pyyaml
   ```

3. Run the collection pipeline using the bundled JSON configuration:

   ```bash
   python scripts/collect_theories.py "activity engagement" --config config/pipeline.json
   ```

   This command reads the seed dataset under `data/examples/`, classifies
   theories, answers the Hackaging Q1–Q9 prompts, and writes CSV exports back to
   the `data/examples/` directory.

4. Refresh question answers and generate an analysis summary:

   ```bash
   python scripts/analyze_papers.py --config config/pipeline.json
   ```

5. Execute the automated tests:

   ```bash
   pytest
   ```

## Configuration

The `config/` directory stores both YAML and JSON versions of the pipeline
configuration. Each file specifies API key placeholders, the seed data source,
keyword maps for theory classification, question-extraction templates, and the
output file locations.

## Sample data

The `data/examples/` folder includes:

- `seed_papers.json` – deterministic seed papers used for tests and demos.
- `papers.csv` – expected schema for exported paper metadata.
- `theories.csv` – sample classifications with theory scores.
- `questions.csv` – answers to Hackaging questions Q1–Q9.

Running the CLI scripts will overwrite these files with fresh outputs generated
from the configured seed data. The latest analysis summary is stored under
`data/cache/analysis_summary.json`.
