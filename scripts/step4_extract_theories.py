"""Derive aging theories mentioned across filtered full-text reviews.

This final step reads the enriched dataset created by
``step3_fetch_fulltext.py`` and prompts an OpenAI model to extract the names of
aging theories discussed in each review. The script aggregates unique theory
labels across all processed articles and stores both the per-article annotations
and the combined set for downstream ontology building. Use ``--processes`` to
control how many worker processes share the LLM queue (auto-detected for large
inputs). The ``--chunk-chars`` and ``--chunk-overlap`` switches control how long
each prompt window is and how much adjacent context overlaps, ensuring long
reviews are streamed to the LLM in multiple passes. The ``--request-timeout``
option controls how long the script waits for each OpenAI API response before
guiding you to retry or restart the step.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for the OpenAI chat completions API.

Usage
-----
```bash
python scripts/step4_extract_theories.py \
    --input data/pipeline/filtered_reviews_fulltext.json \
    --output data/pipeline/aging_theories.json \
    --chunk-chars 12000 --chunk-overlap 1000 \
    --processes 4
```
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import multiprocessing
from multiprocessing.managers import SyncManager
import os
import re
import socket
import sys
import textwrap
import time
import unicodedata
import threading
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import urllib.error
import urllib.request


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_TIMEOUT = 60


PROMPT_LOCK = multiprocessing.Lock()
_NETWORK_ERROR_KEYWORDS = ("network error", "timeout", "temporarily unavailable")


def _is_network_error(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in _NETWORK_ERROR_KEYWORDS)


def _normalise_theory_list(payload: Dict) -> Dict:
    theories = payload.get("theories")
    if theories is None:
        return payload
    if isinstance(theories, list):
        cleaned = []
        for item in theories:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    cleaned.append(name)
        payload["theories"] = cleaned
    elif isinstance(theories, str):
        payload["theories"] = [theories.strip()] if theories.strip() else []
    else:
        payload["theories"] = []
    return payload


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model: str,
    *,
    timeout: Optional[float] = None,
) -> Dict:
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OPENAI_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    request_timeout = DEFAULT_OPENAI_TIMEOUT if timeout is None else timeout

    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network fallback
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API error {exc.code}: {error_body}") from exc
    except (urllib.error.URLError, socket.timeout) as exc:
        raise RuntimeError(
            "OpenAI API request failed due to a network error or timeout. Retry this "
            "step or restart the pipeline."
        ) from exc

    data = json.loads(body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as err:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unexpected OpenAI response: {data!r}") from err

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            "OpenAI returned invalid JSON payload: " + content
        ) from err

    return parsed


def call_openai(
    prompt: str, api_key: str, model: str, *, timeout: Optional[float] = None
) -> Dict:
    system_prompt = (
        "Extract distinct aging theories from the supplied review text. Return "
        "JSON with keys `theories` (list of strings) and `notes` (optional "
        "explanatory text)."
    )
    parsed = chat_completion_json(
        system_prompt, prompt, api_key, model, timeout=timeout
    )
    return _normalise_theory_list(parsed)


def chunk_review_text(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return [""]

    if chunk_chars <= 0 or len(cleaned) <= chunk_chars:
        return [cleaned]

    overlap = max(0, chunk_overlap)
    if overlap >= chunk_chars:
        overlap = max(0, chunk_chars - 1)

    step = max(1, chunk_chars - overlap)
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + chunk_chars)
        chunk = cleaned[start:end]
        if end < length:
            split_point = chunk.rfind(" ")
            if split_point > 0:
                end = start + split_point
                chunk = cleaned[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        else:
            # if whitespace trimming removed content, advance by full window
            chunk = cleaned[start:end]
            if chunk:
                chunks.append(chunk)
        next_start = end if overlap == 0 else max(0, end - overlap)
        if next_start <= start:
            next_start = start + step
        start = next_start
    if not chunks:
        return [cleaned]
    return chunks


def build_prompt(
    record: Dict,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    text = chunk_text or ""
    title = record.get("title", "")
    citation = record.get("journal", "")
    chunk_label = ""
    if total_chunks > 1:
        chunk_label = f" (chunk {chunk_index + 1} of {total_chunks})"
    return textwrap.dedent(
        f"""
        Review title: {title}
        Citation source: {citation}
        Review section{chunk_label}:

        Extract every distinct theory of aging discussed in this review. Focus on
        conceptual theories (e.g., oxidative stress theory, disposable soma,
        network theory). Ignore unrelated topics.

        Review text:
        {text}
        """
    ).strip()


def process_batch(
    payload: Tuple[
        List[Tuple[int, Dict]],
        str,
        str,
        float,
        int,
        int,
        int,
        Optional[float],
        Optional[Any],
    ]
) -> List[Tuple[int, Dict]]:
    (
        batch,
        api_key,
        model,
        delay,
        chunk_chars,
        chunk_overlap,
        total_records,
        request_timeout,
        result_queue,
    ) = payload
    annotated: List[Tuple[int, Dict]] = []
    for global_idx, record in batch:
        text_source = record.get("full_text") or record.get("abstract") or ""
        chunks = chunk_review_text(text_source, chunk_chars, chunk_overlap)
        aggregated_theories: List[str] = []
        seen_aliases: Set[str] = set()
        aggregated_notes: List[str] = []
        for chunk_idx, chunk_text in enumerate(chunks):
            prompt = build_prompt(record, chunk_text, chunk_idx, len(chunks))
            attempt = 0
            backoff_delay = 1.0
            while True:
                try:
                    response = call_openai(
                        prompt, api_key, model, timeout=request_timeout
                    )
                    break
                except RuntimeError as err:
                    message = str(err)
                    if not _is_network_error(message):
                        raise
                    attempt += 1
                    acquired = PROMPT_LOCK.acquire(block=False)
                    if acquired:
                        try:
                            print(
                                (
                                    "Network issue while contacting OpenAI (attempt "
                                    f"{attempt}). Error: {message}\n"
                                    "Verify your internet connection and press Enter to retry."
                                ),
                                flush=True,
                            )
                            try:
                                input("Press Enter once connectivity is restored to retry... ")
                            except EOFError:
                                # In non-interactive environments we still want to pause briefly
                                pass
                        finally:
                            PROMPT_LOCK.release()
                    else:
                        with PROMPT_LOCK:
                            pass
                    sleep_time = min(60.0, backoff_delay)
                    time.sleep(sleep_time)
                    backoff_delay = min(60.0, backoff_delay * 2)
                    continue
            theories = response.get("theories") if isinstance(response, dict) else []
            if isinstance(theories, list):
                for alias in theories:
                    if not isinstance(alias, str):
                        continue
                    cleaned = alias.strip()
                    if not cleaned:
                        continue
                    key = cleaned.casefold()
                    if key not in seen_aliases:
                        seen_aliases.add(key)
                        aggregated_theories.append(cleaned)
            notes_field = response.get("notes") if isinstance(response, dict) else None
            if isinstance(notes_field, str):
                cleaned_note = notes_field.strip()
                if cleaned_note and cleaned_note not in aggregated_notes:
                    aggregated_notes.append(cleaned_note)
            elif isinstance(notes_field, list):
                for note in notes_field:
                    if not isinstance(note, str):
                        continue
                    cleaned_note = note.strip()
                    if cleaned_note and cleaned_note not in aggregated_notes:
                        aggregated_notes.append(cleaned_note)

        annotated_record = record.copy()
        theory_payload: Dict[str, object] = {"theories": aggregated_theories}
        if aggregated_notes:
            theory_payload["notes"] = (
                aggregated_notes[0]
                if len(aggregated_notes) == 1
                else list(aggregated_notes)
            )
        theory_payload["chunk_count"] = len(chunks)
        annotated_record["theory_extraction"] = theory_payload
        annotated.append((global_idx, annotated_record))
        if result_queue is not None:
            result_queue.put((global_idx, annotated_record))
        print(
            f"Annotated {global_idx + 1}/{total_records} reviews",
            flush=True,
        )
        time.sleep(delay)
    return annotated


def extract_theories(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    delay: float,
    chunk_chars: int,
    chunk_overlap: int,
    processes: int,
    request_timeout: Optional[float],
) -> List[Dict]:
    return run_extraction(
        records,
        api_key,
        model,
        delay,
        chunk_chars,
        chunk_overlap,
        processes,
        request_timeout,
        set(),
        None,
    )


def run_extraction(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    delay: float,
    chunk_chars: int,
    chunk_overlap: int,
    processes: int,
    request_timeout: Optional[float],
    processed_indices: Set[int],
    result_queue: Optional[Any],
) -> List[Tuple[int, Dict]]:
    records_list = list(records)
    total_records = len(records_list)
    if total_records == 0:
        return []

    indexed_records: List[Tuple[int, Dict]] = [
        (idx, record)
        for idx, record in enumerate(records_list)
        if idx not in processed_indices
    ]
    if not indexed_records:
        return []
    if processes <= 1:
        payload = (
            indexed_records,
            api_key,
            model,
            delay,
            chunk_chars,
            chunk_overlap,
            total_records,
            request_timeout,
            result_queue,
        )
        annotated_pairs = process_batch(payload)
    else:
        pending_count = len(indexed_records)
        chunk_size = math.ceil(max(1, pending_count) / processes)
        batches: List[List[Tuple[int, Dict]]] = [
            indexed_records[i : i + chunk_size]
            for i in range(0, pending_count, chunk_size)
        ]
        payloads = [
            (
                batch,
                api_key,
                model,
                delay,
                chunk_chars,
                chunk_overlap,
                total_records,
                request_timeout,
                result_queue,
            )
            for batch in batches
        ]
        ctx = multiprocessing.get_context()
        with ctx.Pool(processes=processes) as pool:
            results = pool.map(process_batch, payloads)
        annotated_pairs = [pair for chunk in results for pair in chunk]
        annotated_pairs.sort(key=lambda item: item[0])

    annotated = [record for _, record in annotated_pairs]
    return annotated


def load_checkpoint_annotations(path: str) -> Dict[int, Dict]:
    annotations: Dict[int, Dict] = {}
    if not path or not os.path.exists(path):
        return annotations

    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
    except OSError:
        return annotations

    if not content.strip():
        return annotations

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = payload.get("index")
            record = payload.get("record")
            if isinstance(idx, int) and isinstance(record, dict):
                annotations[idx] = record
        return annotations

    if isinstance(parsed, dict):
        articles = parsed.get("articles") if isinstance(parsed, dict) else None
        if isinstance(articles, list):
            for idx, record in enumerate(articles):
                if isinstance(record, dict):
                    annotations[idx] = record
        try:
            with open(path, "w", encoding="utf-8") as out_fh:
                for idx in sorted(annotations):
                    out_fh.write(
                        json.dumps(
                            {"index": idx, "record": annotations[idx]},
                            ensure_ascii=False,
                        )
                    )
                    out_fh.write("\n")
        except OSError:
            pass
        return annotations

    if isinstance(parsed, list):
        for idx, record in enumerate(parsed):
            if isinstance(record, dict):
                annotations[idx] = record
        try:
            with open(path, "w", encoding="utf-8") as out_fh:
                for idx in sorted(annotations):
                    out_fh.write(
                        json.dumps(
                            {"index": idx, "record": annotations[idx]},
                            ensure_ascii=False,
                        )
                    )
                    out_fh.write("\n")
        except OSError:
            pass
        return annotations

    return annotations


def checkpoint_writer(
    result_queue: "multiprocessing.queues.Queue",  # type: ignore[name-defined]
    path: str,
    annotations: Dict[int, Dict],
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "a", encoding="utf-8") as fh:
        while True:
            item = result_queue.get()
            if item is None:
                break
            index, record = item
            if not isinstance(index, int) or not isinstance(record, dict):
                continue
            if index in annotations:
                continue
            annotations[index] = record
            fh.write(
                json.dumps({"index": index, "record": record}, ensure_ascii=False)
            )
            fh.write("\n")
            fh.flush()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode(
        "ascii"
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def lexical_signature(value: str) -> Tuple[str, Tuple[str, ...]]:
    tokens = re.split(r"[^a-z0-9]+", value.lower())
    tokens = [token for token in tokens if token]
    return (" ".join(tokens), tuple(sorted(tokens)))


def find_lexical_match(
    name: str,
    slug: str,
    slug_index: Dict[str, str],
    signatures: Dict[str, Tuple[str, Tuple[str, ...]]],
) -> Optional[str]:
    if slug and slug in slug_index:
        return slug_index[slug]

    target_signature = lexical_signature(name)
    best_match: Optional[Tuple[str, float]] = None
    for theory_id, signature in signatures.items():
        candidate_text, candidate_tokens = signature
        score = difflib.SequenceMatcher(None, target_signature[0], candidate_text).ratio()
        if score >= 0.9:
            return theory_id
        if target_signature[1] == candidate_tokens and candidate_tokens:
            return theory_id
        if score >= 0.75:
            if best_match is None or score > best_match[1]:
                best_match = (theory_id, score)
    if best_match:
        return best_match[0]
    return None


def disambiguate_with_llm(
    name: str,
    candidates: Dict[str, Dict],
    record: Dict,
    api_key: str,
    model: str,
    request_timeout: Optional[float],
) -> Optional[str]:
    if not candidates:
        return None

    title = record.get("title", "")
    abstract = record.get("abstract", "") or record.get("full_text", "")
    if abstract:
        abstract = abstract[:1500]
    system_prompt = (
        "You reconcile potentially duplicate aging theory names. Analyse the "
        "context and respond with JSON containing keys `match_id` (string or "
        "null) and `rationale` (string). Choose an existing theory ID only if it "
        "clearly refers to the same conceptual theory; otherwise return null."
    )
    candidate_lines = []
    for theory_id, entry in candidates.items():
        label = entry.get("label")
        aliases = ", ".join(entry.get("aliases", [])) or "(no aliases yet)"
        candidate_lines.append(f"- {theory_id}: {label} | aliases: {aliases}")
    user_prompt = textwrap.dedent(
        f"""
        New theory mention: {name}
        Article title: {title}

        Known candidates:
        {os.linesep.join(candidate_lines)}

        Article context snippet:
        {abstract}
        """
    ).strip()
    try:
        response = chat_completion_json(
            system_prompt,
            user_prompt,
            api_key,
            model,
            timeout=request_timeout,
        )
    except RuntimeError:
        return None
    match_id = response.get("match_id")
    if isinstance(match_id, str) and match_id in candidates:
        rationale = response.get("rationale")
        notes = candidates[match_id].setdefault("notes", [])
        if isinstance(rationale, str) and rationale.strip():
            notes.append(rationale.strip())
        return match_id
    return None


def build_theory_registry(
    annotated: Iterable[Dict],
    api_key: str,
    model: str,
    request_timeout: Optional[float],
) -> Dict[str, Dict]:
    registry: Dict[str, Dict] = {}
    signature_index: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
    slug_index: Dict[str, str] = {}

    def add_alias(entry: Dict, alias: str) -> None:
        aliases: Set[str] = set(entry.setdefault("aliases", []))
        if alias not in aliases:
            aliases.add(alias)
            entry["aliases"] = sorted(aliases)

    for idx, record in enumerate(annotated):
        extraction = record.get("theory_extraction") or {}
        theories = extraction.get("theories") or []
        if not isinstance(theories, list):
            continue
        article_assignments: List[Dict[str, str]] = []

        article_id = record.get("id") or record.get("uid") or record.get("doi")
        if not article_id:
            article_id = record.get("openalex_id") or record.get("pmid")
        if not article_id:
            article_id = f"article-{idx+1}"
        openalex_id = record.get("openalex_id")
        pmid = record.get("pmid")

        for raw_name in theories:
            if not isinstance(raw_name, str):
                continue
            cleaned = raw_name.strip()
            if not cleaned:
                continue
            slug = slugify(cleaned)
            signature = lexical_signature(cleaned)
            theory_id = None

            # direct slug match
            if slug and slug in slug_index:
                theory_id = slug_index[slug]
            else:
                candidate_id = find_lexical_match(
                    cleaned, slug, slug_index, signature_index
                )
                if candidate_id and candidate_id in registry:
                    theory_id = candidate_id
                else:
                    potential_candidates = {
                        key: value
                        for key, value in registry.items()
                        if difflib.SequenceMatcher(
                            None,
                            signature[0],
                            signature_index.get(key, ("", tuple()))[0],
                        ).ratio()
                        >= 0.65
                    }
                    match = disambiguate_with_llm(
                        cleaned,
                        potential_candidates,
                        record,
                        api_key,
                        model,
                        request_timeout,
                    )
                    if match:
                        theory_id = match

            if theory_id is None:
                provenance_bits: List[str] = []
                if openalex_id:
                    provenance_bits.append(str(openalex_id))
                if pmid:
                    provenance_bits.append(str(pmid))
                provenance_bits.append(article_id)
                provenance = "-".join(provenance_bits)
                theory_id = f"{slug or 'theory'}__{slugify(provenance)}"
                registry[theory_id] = {
                    "label": cleaned,
                    "slug": slug,
                    "provenance": {
                        "article_id": article_id,
                        "openalex_id": openalex_id,
                        "pmid": pmid,
                    },
                    "supporting_articles": [],
                    "openalex_ids": [],
                    "pmids": [],
                    "aliases": [],
                    "notes": [],
                }
                signature_index[theory_id] = signature
                if slug:
                    slug_index.setdefault(slug, theory_id)
            else:
                # update slug index if new slug was unseen
                if slug and slug not in slug_index:
                    slug_index[slug] = theory_id
                signature_index.setdefault(theory_id, signature)

            entry = registry[theory_id]
            add_alias(entry, cleaned)
            entry.setdefault("supporting_articles", [])
            if article_id not in entry["supporting_articles"]:
                entry["supporting_articles"].append(article_id)
            if openalex_id:
                entry.setdefault("openalex_ids", [])
                if openalex_id not in entry["openalex_ids"]:
                    entry["openalex_ids"].append(openalex_id)
            if pmid:
                entry.setdefault("pmids", [])
                if pmid not in entry["pmids"]:
                    entry["pmids"].append(pmid)
            notes_field = extraction.get("notes")
            note_values: List[str] = []
            if isinstance(notes_field, str):
                cleaned_note = notes_field.strip()
                if cleaned_note:
                    note_values.append(cleaned_note)
            elif isinstance(notes_field, list):
                for note in notes_field:
                    if not isinstance(note, str):
                        continue
                    cleaned_note = note.strip()
                    if cleaned_note:
                        note_values.append(cleaned_note)
            if note_values:
                notes = entry.setdefault("notes", [])
                for note in note_values:
                    if note not in notes:
                        notes.append(note)

            article_assignments.append({"theory_id": theory_id, "alias": cleaned})

        if article_assignments:
            record["theory_assignments"] = article_assignments

    for entry in registry.values():
        entry.setdefault("aliases", [])
        entry["aliases"] = sorted(dict.fromkeys(entry["aliases"]))
        entry.setdefault("supporting_articles", [])
        entry["supporting_articles"] = sorted(entry["supporting_articles"])
        entry.setdefault("openalex_ids", [])
        entry["openalex_ids"] = sorted(entry["openalex_ids"])
        entry.setdefault("pmids", [])
        entry["pmids"] = sorted(entry["pmids"])

    return registry


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract aging theories from filtered reviews")
    parser.add_argument("--input", default="data/pipeline/filtered_reviews_fulltext.json")
    parser.add_argument("--output", default="data/pipeline/aging_theories.json")
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help=(
            "OpenAI chat completion model identifier. gpt-5-nano is the default "
            "so consolidated review excerpts can be processed accurately "
            "without exceeding the ~$10 per million articles budget."
        ),
    )
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=12000,
        help=(
            "Maximum number of characters in each review chunk sent to the LLM."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=1000,
        help=(
            "Number of characters to overlap between successive review chunks when "
            "calling the LLM."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT,
        help=(
            "Timeout (in seconds) for each OpenAI API request before retry guidance "
            "is raised."
        ),
    )
    parser.add_argument(
        "--max-chars",
        dest="compat_max_chars",
        type=int,
        default=None,
        help="Deprecated alias for --chunk-chars.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Number of worker processes to use for LLM calls; when omitted the script "
            "auto-scales using the available CPU cores while keeping roughly 25 "
            "reviews per worker."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a checkpoint file used to persist per-article annotations. "
            "Defaults to the --output path."
        ),
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        print("Input JSON must contain a list of records", file=sys.stderr)
        return 1

    cpu_total = os.cpu_count() or 1
    min_records_per_worker = 25
    if len(records):
        auto_processes = min(cpu_total, max(1, math.ceil(len(records) / min_records_per_worker)))
    else:
        auto_processes = 1
    processes = args.processes or auto_processes
    if len(records) == 0:
        processes = 0
    else:
        processes = max(1, min(processes, len(records)))

    chunk_chars = args.chunk_chars
    if args.compat_max_chars is not None:
        chunk_chars = args.compat_max_chars

    checkpoint_path = args.checkpoint or args.output
    checkpoint_annotations = load_checkpoint_annotations(checkpoint_path)
    total_records = len(records)
    processed_indices = {
        idx for idx in checkpoint_annotations.keys() if 0 <= idx < total_records
    }
    if processed_indices:
        print(
            f"Loaded {len(processed_indices)} existing annotations from {checkpoint_path}",
            flush=True,
        )

    pending_indices = [idx for idx in range(total_records) if idx not in processed_indices]

    queue_ctx = multiprocessing.get_context()
    manager: Optional[SyncManager] = None
    result_queue = None
    writer_thread: Optional[threading.Thread] = None
    if pending_indices:
        manager = queue_ctx.Manager()
        result_queue = manager.Queue()
        writer_thread = threading.Thread(
            target=checkpoint_writer,
            args=(result_queue, checkpoint_path, checkpoint_annotations),
            daemon=True,
        )
        writer_thread.start()
        try:
            run_extraction(
                records,
                api_key,
                args.model,
                args.delay,
                chunk_chars,
                args.chunk_overlap,
                processes,
                args.request_timeout,
                processed_indices,
                result_queue,
            )
        finally:
            if result_queue is not None:
                try:
                    result_queue.put(None)
                except Exception:
                    pass
            if writer_thread is not None:
                writer_thread.join(timeout=30)
                if writer_thread.is_alive():
                    print(
                        "Timed out waiting for checkpoint writer to finish; continuing without"
                        " guaranteed checkpoint flush.",
                        flush=True,
                    )
            if result_queue is not None:
                close = getattr(result_queue, "close", None)
                join_thread = getattr(result_queue, "join_thread", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
                if callable(join_thread):
                    try:
                        join_thread()
                    except Exception:
                        pass
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
    else:
        print("All records already processed; skipping extraction phase.", flush=True)

    annotations_after_run = load_checkpoint_annotations(checkpoint_path)
    annotated = [annotations_after_run[idx] for idx in sorted(annotations_after_run)]

    theory_registry = build_theory_registry(
        annotated, api_key, args.model, args.request_timeout
    )
    aggregated = sorted(theory_registry.keys())

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "articles": annotated,
                "aggregated_theories": aggregated,
                "theory_registry": theory_registry,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved theory catalogue with {len(theory_registry)} canonical theories to {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

