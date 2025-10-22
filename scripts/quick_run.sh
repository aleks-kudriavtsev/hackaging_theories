#!/usr/bin/env bash
set -euo pipefail
WORKDIR="${1:-data/pipeline}"
FILTER_PROCS="${2:-6}"
CLASSIFY_WORKERS="${3:-4}"
THEORY_WORKERS="${4:-4}"
DELAY="${5:-1}"   # delay between batches (seconds)

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: set OPENAI_API_KEY env var before running."
  exit 1
fi

echo "Quick run: workdir=$WORKDIR filter_procs=$FILTER_PROCS classify_workers=$CLASSIFY_WORKERS theory_workers=$THEORY_WORKERS"
echo "Models: filter=gpt-5-nano, classify=gpt-5-nano, extract=gpt-5-mini"

# Step 1: bootstrap / collect initial reviews
echo "STEP 1: bootstrap collect reviews..."
python scripts/step1_collect_reviews.py --workdir "$WORKDIR" --max 200 || { echo "step1 failed"; exit 1; }

# Step 2: filter reviews (fast model)
echo "STEP 2: filter reviews (gpt-5-nano) with $FILTER_PROCS processes..."
python scripts/step2_filter_reviews.py --workdir "$WORKDIR" --processes "$FILTER_PROCS" --model gpt-5-nano --delay "$DELAY" || { echo "step2 failed"; exit 1; }

# Step 3: fetch fulltexts (parallel network-bound)
echo "STEP 3: fetch fulltexts..."
python scripts/step3_fetch_fulltext.py --workdir "$WORKDIR" --threads 8 || { echo "step3 failed"; exit 1; }

# Step 4: extract theories from fulltext (use gpt-5-nano for accuracy; few calls)
echo "STEP 4: extract theories (gpt-5-nano)"
python scripts/step4_extract_theories.py --workdir "$WORKDIR" --processes "$THEORY_WORKERS" --model gpt-5-nano || { echo "step4 failed"; exit 1; }

# Step 5: build ontology (local grouping, light LLM)
echo "STEP 5: build ontology..."
python scripts/step5_build_ontology.py --workdir "$WORKDIR" || { echo "step5 failed"; exit 1; }

# Step 6: collect and classify papers per theory + answer Q1..Q9 (bulk, use gpt-5-nano)
echo "STEP 6: collect and classify; this is the longest step..."
python scripts/collect_theories.py --workdir "$WORKDIR" --classification-workers "$CLASSIFY_WORKERS" --model gpt-5-nano --delay "$DELAY" --resume || { echo "step6 failed"; exit 1; }

echo "Pipeline finished. Outputs in $WORKDIR/competition/ (theories.csv, theory_papers.csv, questions.csv)"
