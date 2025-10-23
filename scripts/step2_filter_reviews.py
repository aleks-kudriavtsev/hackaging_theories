"""Filter and enrich aging-theory reviews in a single pass.

The script ingests the metadata collected during step 1, calls an OpenAI chat
model to flag reviews that genuinely discuss aging theory, and immediately
retrieves accessible full texts for the retained entries (PMC XML when
available, otherwise OpenAlex PDFs). The resulting documents are chunked and
sent through the gpt-5-nano extraction routines so raw theory mentions, their
supporting article IDs and a canonical registry are produced in the same run.
Large batches are automatically processed in parallel worker processes so long
runs saturate the available CPU cores.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for calling the chat completion endpoint.
- ``PUBMED_RATE_INTERVAL`` / ``PUBMED_MAX_ATTEMPTS`` / ``PUBMED_RETRY_WAIT`` /
  ``PUBMED_BATCH_SIZE`` — optional overrides for the Entrez/PubMed rate
  limiting parameters reused from :mod:`step3_fetch_fulltext`.

Usage
-----
```bash
python scripts/step2_filter_reviews.py \
    --input data/pipeline/start_reviews.json \
    --filtered-output data/pipeline/filtered_reviews.json \
    --output data/pipeline/aging_theories.json \
    --processes 4 --fulltext-processes 4 --extraction-processes 4
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from openai import AsyncOpenAI

try:
    from scripts.step3_fetch_fulltext import enrich_records
    from scripts.step4_extract_theories import (
        DEFAULT_OPENAI_TIMEOUT,
        build_theory_registry,
        extract_theories,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.step3_fetch_fulltext import enrich_records
    from scripts.step4_extract_theories import (
        DEFAULT_OPENAI_TIMEOUT,
        build_theory_registry,
        extract_theories,
    )


OPENAI_SYSTEM_PROMPT = (
    "You are an expert gerontology curator. Respond with JSON containing `relevant` "
    "(true/false) and `explanation`."
)


def _coerce_bool(value: object) -> bool | None:
    """Interpret truthy strings/ints returned by the model."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


async def _call_openai(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> Dict:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - network error handling
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
        finally:
            if delay > 0:
                await asyncio.sleep(delay)

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as err:  # pragma: no cover
        raise RuntimeError(f"Unexpected OpenAI response: {response!r}") from err

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            "OpenAI returned invalid JSON payload: " + content
        ) from err

    return parsed


def call_openai(prompt: str, api_key: str, model: str) -> Dict:
    """Backward-compatible synchronous wrapper around :func:`_call_openai`."""

    async def _run() -> Dict:
        async with AsyncOpenAI(api_key=api_key) as client:  # type: ignore[call-arg]
            semaphore = asyncio.Semaphore(1)
            return await _call_openai(client, prompt, model, semaphore, delay=0)

    return asyncio.run(_run())


def build_prompt(batch: Sequence[Dict]) -> str:
    lines = [
        "You will receive multiple review articles about scientific topics.",
        "For each item, determine if the article is primarily about aging theory.",
        "Return a JSON array with the same number of elements as the items below.",
        "Each array element must correspond to the item at the same position.",
        "Every element must be an object with fields `relevant` (true/false) and `explanation`.",
        "Items:",
    ]

    for index, record in enumerate(batch, start=1):
        abstract = record.get("abstract") or ""
        sources = record.get("sources")
        source_line = ""
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            joined = ", ".join(
                str(item) for item in sources if isinstance(item, str) and item.strip()
            )
            if joined:
                source_line = f"Sources: {joined}\n"
        elif isinstance(sources, str) and sources.strip():
            source_line = f"Sources: {sources.strip()}\n"
        elif isinstance(record.get("provenance"), str) and record["provenance"].strip():
            source_line = f"Sources: {record['provenance'].strip()}\n"

        lines.extend(
            [
                f"Item {index}:",
                f"Title: {record.get('title', '')}",
                source_line.rstrip(),
                f"Abstract: {abstract}",
                "",
            ]
        )

    return "\n".join(line for line in lines if line is not None)


def _normalize_item(item: object) -> Dict | None:
    if not isinstance(item, dict):
        return None

    relevant = _coerce_bool(item.get("relevant"))
    if relevant is None:
        return None

    normalized = dict(item)
    normalized["relevant"] = relevant
    if "explanation" not in normalized or normalized["explanation"] is None:
        normalized["explanation"] = ""

    return normalized


def _parse_batch_response(
    batch: Sequence[Dict], response: object
) -> Tuple[Dict[int, Dict], List[int]]:
    processed: Dict[int, Dict] = {}
    fallback: List[int] = []

    if isinstance(response, list):
        elements = response
        if len(elements) != len(batch):
            # keep len(elements) so we don't index past range
            elements = list(elements)[: len(batch)]
            fallback.extend(range(len(elements), len(batch)))
        for idx, item in enumerate(elements):
            if not isinstance(item, dict):
                fallback.append(idx)
                continue

            relevant = _coerce_bool(item.get("relevant"))
            if relevant is None:
                fallback.append(idx)
                continue

            item_data = dict(item)
            item_data["relevant"] = relevant
            if "explanation" not in item_data:
                item_data["explanation"] = ""
            processed[idx] = item_data

        missing = set(range(len(batch))) - set(processed.keys()) - set(fallback)
        if missing:
            fallback.extend(sorted(missing))

        return processed, sorted(set(fallback))

    if not isinstance(response, dict):
        return processed, list(range(len(batch)))

    if len(batch) == 1:
        single_item = _normalize_item(response)
        if single_item is not None:
            processed[0] = single_item
            return processed, []

    for idx in range(len(batch)):
        key = str(idx + 1)
        item_data = _normalize_item(response.get(key))
        if item_data is None:
            fallback.append(idx)
            continue

        processed[idx] = item_data

    missing = set(range(len(batch))) - set(processed.keys()) - set(fallback)
    if missing:
        fallback.extend(sorted(missing))

    return processed, sorted(set(fallback))


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = doi.strip().lower()
    if not cleaned:
        return None
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.strip() or None


def _record_cache_keys(record: Mapping[str, object]) -> List[str]:
    keys: List[str] = []

    doi_value = record.get("doi")
    if isinstance(doi_value, str):
        normalised = _normalise_doi(doi_value)
        if normalised:
            keys.append(f"doi:{normalised}")

    pmid = record.get("pmid")
    if isinstance(pmid, str):
        cleaned = pmid.strip()
        if cleaned:
            keys.append(f"pmid:{cleaned}")

    openalex = record.get("openalex_id")
    if isinstance(openalex, str):
        cleaned = openalex.strip()
        if cleaned:
            keys.append(f"openalex:{cleaned}")

    generic_id = record.get("id")
    if isinstance(generic_id, str):
        cleaned = generic_id.strip()
        if cleaned:
            keys.append(f"id:{cleaned.lower()}")

    title = record.get("title")
    if isinstance(title, str):
        cleaned = " ".join(title.split()).lower()
        if cleaned:
            year = record.get("publication_year")
            if isinstance(year, int):
                keys.append(f"title:{cleaned}|year:{year}")
            elif isinstance(year, str) and year.strip():
                keys.append(f"title:{cleaned}|year:{year.strip()}")
            keys.append(f"title:{cleaned}")

    if not keys:
        abstract = record.get("abstract")
        if isinstance(abstract, str):
            cleaned = " ".join(abstract.split()).lower()
            if cleaned:
                keys.append(f"abstract:{cleaned[:200]}")

    return keys


def _stringify_identifier(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _stable_identifier_keys(record: Mapping[str, object] | None) -> List[str]:
    if not isinstance(record, Mapping):
        return []

    keys: List[str] = []

    doi_value = record.get("doi")
    if isinstance(doi_value, str):
        normalised = _normalise_doi(doi_value)
        if normalised:
            keys.append(f"doi:{normalised}")

    pmid_value = record.get("pmid")
    if pmid_value is not None:
        cleaned = _stringify_identifier(pmid_value)
        if cleaned:
            keys.append(f"pmid:{cleaned}")

    openalex_fields = ("openalex_id", "openalex")
    for field in openalex_fields:
        value = record.get(field)
        if value is None:
            continue
        cleaned = _stringify_identifier(value)
        if cleaned:
            keys.append(f"openalex:{cleaned}")
            break

    pmcid_value = record.get("pmcid")
    if pmcid_value is not None:
        cleaned = _stringify_identifier(pmcid_value)
        if cleaned:
            keys.append(f"pmcid:{cleaned}")

    return keys


def _load_failure_log(path: str | os.PathLike[str] | None) -> List[Dict]:
    if not path:
        return []

    failure_path = Path(path)
    if not failure_path.exists():
        return []

    try:
        payload = json.loads(failure_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"Warning: could not decode failure log at {failure_path}: {err}", file=sys.stderr)
        return []

    if isinstance(payload, list):
        failures: List[Dict] = []
        for item in payload:
            if isinstance(item, Mapping):
                failures.append(dict(item))
        return failures

    return []


def _index_existing_payload(
    payload: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    articles_by_identifier: Dict[str, Dict[str, Any]] = {}
    articles_by_id: Dict[str, Dict[str, Any]] = {}
    raw_mentions_by_identifier: Dict[str, List[Dict[str, Any]]] = {}
    registry_by_identifier: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    synonym_by_identifier: Dict[str, List[Dict[str, Any]]] = {}

    articles = payload.get("articles")
    if isinstance(articles, Sequence) and not isinstance(articles, (str, bytes)):
        for idx, article in enumerate(articles):
            if not isinstance(article, Mapping):
                continue
            article_dict = dict(article)
            article_id = article_dict.get("article_id")
            if not isinstance(article_id, str) or not article_id.strip():
                article_id = _resolve_article_id(article_dict, idx)
            else:
                article_id = article_id.strip()
            article_dict.setdefault("article_id", article_id)
            articles_by_id[article_id] = article_dict
            for key in _stable_identifier_keys(article_dict):
                articles_by_identifier[key] = article_dict

    raw_mentions = payload.get("raw_theory_mentions")
    if isinstance(raw_mentions, Sequence) and not isinstance(raw_mentions, (str, bytes)):
        for mention in raw_mentions:
            if not isinstance(mention, Mapping):
                continue
            mention_dict = dict(mention)
            mention_keys = _stable_identifier_keys(mention_dict)
            article_id = mention_dict.get("article_id")
            if isinstance(article_id, str) and article_id in articles_by_id:
                article_keys = _stable_identifier_keys(articles_by_id[article_id])
                mention_keys.extend(article_keys)
            for key in mention_keys:
                if not key:
                    continue
                raw_mentions_by_identifier.setdefault(key, []).append(mention_dict)

    theory_registry = payload.get("theory_registry")
    if isinstance(theory_registry, Mapping):
        for theory_id, entry in theory_registry.items():
            if not isinstance(entry, Mapping):
                continue
            entry_dict = dict(entry)
            related_keys: List[str] = []
            for key in entry_dict.get("pmids", []) or []:
                cleaned = _stringify_identifier(key)
                if cleaned:
                    related_keys.append(f"pmid:{cleaned}")
            for key in entry_dict.get("openalex_ids", []) or []:
                cleaned = _stringify_identifier(key)
                if cleaned:
                    related_keys.append(f"openalex:{cleaned}")
            provenance = entry_dict.get("provenance")
            if isinstance(provenance, Mapping):
                related_keys.extend(_stable_identifier_keys(provenance))
                article_id = provenance.get("article_id")
                if isinstance(article_id, str) and article_id in articles_by_id:
                    related_keys.extend(_stable_identifier_keys(articles_by_id[article_id]))
            supporting = entry_dict.get("supporting_articles")
            if isinstance(supporting, Sequence) and not isinstance(supporting, (str, bytes)):
                for article_id in supporting:
                    if not isinstance(article_id, str):
                        continue
                    article = articles_by_id.get(article_id)
                    if article:
                        related_keys.extend(_stable_identifier_keys(article))
            for key in related_keys:
                if not key:
                    continue
                registry_by_identifier.setdefault(key, []).append((str(theory_id), entry_dict))

    synonym_registry = payload.get("synonym_registry")
    if isinstance(synonym_registry, Mapping):
        for alias_key, record in synonym_registry.items():
            if not isinstance(record, Mapping):
                continue
            record_dict = dict(record)
            source_articles = record_dict.get("source_articles")
            related_keys: List[str] = []
            if isinstance(source_articles, Sequence) and not isinstance(source_articles, (str, bytes)):
                for article_id in source_articles:
                    if not isinstance(article_id, str):
                        continue
                    article = articles_by_id.get(article_id)
                    if article:
                        related_keys.extend(_stable_identifier_keys(article))
            provenance = record_dict.get("provenance")
            if isinstance(provenance, Mapping):
                related_keys.extend(_stable_identifier_keys(provenance))
            for key in related_keys:
                if not key:
                    continue
                synonym_by_identifier.setdefault(key, []).append(record_dict)

    return {
        "articles_by_identifier": articles_by_identifier,
        "articles_by_id": articles_by_id,
        "raw_mentions_by_identifier": raw_mentions_by_identifier,
        "registry_by_identifier": registry_by_identifier,
        "synonyms_by_identifier": synonym_by_identifier,
    }


def _load_existing_payload(
    path: str | os.PathLike[str] | None,
) -> Tuple[Mapping[str, Any], Dict[str, Dict[str, Any]]]:
    if not path:
        return {}, {
            "articles_by_identifier": {},
            "articles_by_id": {},
            "raw_mentions_by_identifier": {},
            "registry_by_identifier": {},
            "synonyms_by_identifier": {},
        }

    output_path = Path(path)
    if not output_path.exists():
        return {}, {
            "articles_by_identifier": {},
            "articles_by_id": {},
            "raw_mentions_by_identifier": {},
            "registry_by_identifier": {},
            "synonyms_by_identifier": {},
        }

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"Warning: could not decode existing payload at {output_path}: {err}", file=sys.stderr)
        return {}, {
            "articles_by_identifier": {},
            "articles_by_id": {},
            "raw_mentions_by_identifier": {},
            "registry_by_identifier": {},
            "synonyms_by_identifier": {},
        }

    if not isinstance(payload, Mapping):
        return {}, {
            "articles_by_identifier": {},
            "articles_by_id": {},
            "raw_mentions_by_identifier": {},
            "registry_by_identifier": {},
            "synonyms_by_identifier": {},
        }

    indexes = _index_existing_payload(payload)
    return payload, indexes


def _failure_identifier_keys(failure: Mapping[str, Any]) -> List[str]:
    keys = []
    for key in _stable_identifier_keys(failure):
        keys.append(key)
    url = failure.get("url") if isinstance(failure, Mapping) else None
    if isinstance(url, str) and url.strip():
        keys.append(f"url:{url.strip()}")
    return keys


def _merge_failures(
    existing: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
    success_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    success_keys: set[str] = set()
    for record in success_records:
        for key in _stable_identifier_keys(record):
            success_keys.add(key)

    merged: List[Dict[str, Any]] = []
    key_to_index: Dict[str, int] = {}

    def add_failure(entry: Mapping[str, Any]) -> None:
        if not isinstance(entry, Mapping):
            return
        keys = _failure_identifier_keys(entry)
        if keys:
            stable_subset = [key for key in keys if key.split(":", 1)[0] in {"pmid", "doi", "openalex"}]
            if any(key in success_keys for key in stable_subset):
                return
        target_index: Optional[int] = None
        for key in keys:
            existing_index = key_to_index.get(key)
            if existing_index is not None:
                target_index = existing_index
                break
        entry_dict = dict(entry)
        if target_index is not None:
            merged[target_index] = entry_dict
        else:
            merged.append(entry_dict)
            target_index = len(merged) - 1
        for key in keys:
            key_to_index[key] = target_index

    for failure in existing:
        add_failure(failure)

    for failure in new:
        add_failure(failure)

    return merged


def load_decision_cache(path: str | os.PathLike[str] | None) -> Dict[str, Dict]:
    if not path:
        return {}

    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        print(f"Warning: could not decode cache at {cache_path}: {err}", file=sys.stderr)
        return {}

    cache: Dict[str, Dict] = {}
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            normalized = _normalize_item(value)
            if normalized is not None:
                cache[key] = normalized

    return cache


async def async_filter_records(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    *,
    delay: float = 0.5,
    batch_size: int = 5,
    concurrency: int = 5,
    progress_prefix: str | None = None,
    cache: MutableMapping[str, Dict] | None = None,
    cache_path: str | os.PathLike[str] | None = None,
) -> List[Dict]:
    records_list = list(records)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    cache_map: MutableMapping[str, Dict]
    if cache is None:
        cache_map = {}
    else:
        cache_map = cache

    cache_path_str = str(cache_path) if cache_path is not None else None

    record_keys_list = [_record_cache_keys(record) for record in records_list]
    pending_records: List[Dict] = []
    pending_keys: List[List[str]] = []
    pending_original_indices: List[int] = []
    cached_hits: List[int] = []
    cached_updates: List[Tuple[List[str], Dict]] = []

    for idx, record in enumerate(records_list):
        keys = record_keys_list[idx]
        decision: Dict | None = None
        for key in keys:
            if key and key in cache_map:
                decision = dict(cache_map[key])
                break
        if decision is not None:
            record["llm_filter"] = dict(decision)
            cached_hits.append(idx)
            missing = [key for key in keys if key and key not in cache_map]
            if missing:
                cached_updates.append((missing, dict(decision)))
        else:
            pending_records.append(record)
            pending_keys.append(keys)
            pending_original_indices.append(idx)

    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    cache_write_lock = asyncio.Lock()
    processed_total = 0
    kept_total = 0

    async with AsyncOpenAI(api_key=api_key) as client:
        prefix = f"[{progress_prefix}] " if progress_prefix else ""

        async def log_progress(relevant: bool) -> None:
            nonlocal processed_total, kept_total
            async with progress_lock:
                processed_total += 1
                if relevant:
                    kept_total += 1
                print(
                    f"{prefix}Processed {processed_total} records — kept {kept_total}",
                    flush=True,
                )

        async def persist_entries(entries: Sequence[Tuple[Sequence[str], Mapping[str, object]]]) -> None:
            if not entries:
                return
            async with cache_write_lock:
                changed = False
                for keys, decision in entries:
                    normalized = _normalize_item(dict(decision))
                    if normalized is None:
                        continue
                    stored = dict(normalized)
                    for key in keys:
                        if not key:
                            continue
                        existing = cache_map.get(key)
                        if existing == stored:
                            continue
                        cache_map[key] = stored
                        changed = True
                if changed and cache_path_str:
                    cache_dir = os.path.dirname(cache_path_str)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    tmp_path = cache_path_str + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as handle:
                        json.dump(cache_map, handle, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, cache_path_str)

        if cached_updates:
            await persist_entries(cached_updates)

        if cached_hits:
            print(
                f"{prefix}Reused cached decisions for {len(cached_hits)} records",
                flush=True,
            )
            for idx in cached_hits:
                decision = records_list[idx].get("llm_filter", {})
                relevant = False
                if isinstance(decision, Mapping):
                    relevant = decision.get("relevant") is True
                await log_progress(relevant)

        if not pending_records:
            return [
                record
                for record in records_list
                if record.get("llm_filter", {}).get("relevant") is True
            ]

        async def process_batch(start: int) -> None:
            batch_records = pending_records[start : start + batch_size]
            batch_keys = pending_keys[start : start + batch_size]
            batch_indices = pending_original_indices[start : start + batch_size]
            prompt = build_prompt(batch_records)
            processed_items: Dict[int, Dict] = {}
            fallback_indices: List[int] = []
            try:
                response = await _call_openai(
                    client,
                    prompt,
                    model,
                    semaphore,
                    delay,
                )
            except RuntimeError as err:
                if batch_indices:
                    first = batch_indices[0] + 1
                    last = batch_indices[-1] + 1
                    range_desc = f"{first}-{last}" if first != last else str(first)
                else:
                    range_desc = str(start + 1)
                print(
                    f"{prefix}Batch {range_desc} failed with error: {err}. Retrying items individually.",
                    file=sys.stderr,
                )
                fallback_indices = list(range(len(batch_records)))
            else:
                processed_items, fallback_indices = _parse_batch_response(batch_records, response)

            batch_updates: List[Tuple[List[str], Dict]] = []

            for offset, decision in processed_items.items():
                record = batch_records[offset]
                record["llm_filter"] = decision
                relevant = decision.get("relevant") is True
                await log_progress(relevant)
                keys = [key for key in batch_keys[offset] if key]
                if keys:
                    batch_updates.append((keys, decision))

            for offset in fallback_indices:
                record = batch_records[offset]
                single_prompt = build_prompt([record])
                try:
                    single_response = await _call_openai(
                        client,
                        single_prompt,
                        model,
                        semaphore,
                        delay,
                    )
                except RuntimeError as err:
                    record_number = batch_indices[offset] + 1 if offset < len(batch_indices) else start + offset + 1
                    print(
                        f"{prefix}Record {record_number} failed after retry: {err}",
                        file=sys.stderr,
                    )
                    await log_progress(False)
                    continue

                single_items, pending = _parse_batch_response([record], single_response)
                decision = single_items.get(0)
                relevant = False
                if decision is not None:
                    record["llm_filter"] = decision
                    relevant = decision.get("relevant") is True
                    keys = [key for key in batch_keys[offset] if key]
                    if keys:
                        batch_updates.append((keys, decision))
                else:
                    record_number = batch_indices[offset] + 1 if offset < len(batch_indices) else start + offset + 1
                    print(
                        f"{prefix}Record {record_number} returned invalid JSON even after retry.",
                        file=sys.stderr,
                    )
                if pending:
                    record_number = batch_indices[offset] + 1 if offset < len(batch_indices) else start + offset + 1
                    print(
                        f"{prefix}Record {record_number} still pending after retry; giving up.",
                        file=sys.stderr,
                    )
                await log_progress(relevant)

            if batch_updates:
                await persist_entries(batch_updates)

        tasks = [
            asyncio.create_task(process_batch(start))
            for start in range(0, len(pending_records), batch_size)
        ]
        if tasks:
            await asyncio.gather(*tasks)

    kept = [
        record
        for record in records_list
        if record.get("llm_filter", {}).get("relevant") is True
    ]
    return kept


def filter_records(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    *,
    delay: float = 0.5,
    batch_size: int = 5,
    concurrency: int = 5,
    cache: MutableMapping[str, Dict] | None = None,
    cache_path: str | os.PathLike[str] | None = None,
) -> List[Dict]:
    """Synchronous helper mirroring :func:`async_filter_records`.

    The implementation swaps the internal coroutine transport so legacy callers
    that monkeypatch :func:`call_openai` continue to work. The heavy lifting is
    still delegated to :func:`async_filter_records` so behaviour stays aligned
    between both entry points.
    """

    async def _compat_call_openai(
        client: AsyncOpenAI,
        prompt: str,
        model_name: str,
        semaphore: asyncio.Semaphore,
        call_delay: float,
    ) -> Dict:
        async with semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, call_openai, prompt, api_key, model_name
            )
            if call_delay > 0:
                await asyncio.sleep(call_delay)
            return result

    original_call = _call_openai
    try:
        globals()["_call_openai"] = _compat_call_openai
        return asyncio.run(
            async_filter_records(
                records,
                api_key,
                model,
                delay=delay,
                batch_size=batch_size,
                concurrency=concurrency,
                cache=cache,
                cache_path=cache_path,
            )
        )
    finally:
        globals()["_call_openai"] = original_call


def _split_records(records: List[Dict], parts: int) -> List[List[Dict]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if not records:
        return [[] for _ in range(parts)]

    chunk_sizes = []
    base, remainder = divmod(len(records), parts)
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        chunk_sizes.append(size)

    chunks: List[List[Dict]] = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(list(records[start:end]))
        start = end

    return chunks


def _resolve_article_id(record: Mapping[str, object], index: int) -> str:
    explicit = record.get("article_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    preferred_keys = ("id", "uid", "doi")
    fallback_keys = ("openalex_id", "pmid")

    for key in (*preferred_keys, *fallback_keys):
        value = record.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        elif value is not None:
            stringified = str(value).strip()
            if stringified:
                return stringified

    return f"article-{index + 1}"


def _collect_theory_mentions(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        article_id = _resolve_article_id(record, idx)
        if isinstance(record, dict) and "article_id" not in record:
            record["article_id"] = article_id

        extraction = record.get("theory_extraction")
        theories = extraction.get("theories") if isinstance(extraction, Mapping) else None
        if not isinstance(theories, Sequence) or isinstance(theories, (str, bytes)):
            continue

        for alias in theories:
            if not isinstance(alias, str):
                continue
            cleaned = alias.strip()
            if not cleaned:
                continue
            mention: Dict[str, Any] = {
                "article_id": article_id,
                "alias": cleaned,
                "record_index": idx,
            }
            title = record.get("title")
            if isinstance(title, str) and title.strip():
                mention["title"] = title.strip()
            openalex_id = record.get("openalex_id")
            if isinstance(openalex_id, str) and openalex_id.strip():
                mention["openalex_id"] = openalex_id.strip()
            pmid = record.get("pmid")
            if isinstance(pmid, str) and pmid.strip():
                mention["pmid"] = pmid.strip()
            mentions.append(mention)

    return mentions


def _worker_filter(
    index: int,
    records: List[Dict],
    queue: "Queue[Tuple[int, List[Dict] | None, str | None]]",
    api_key: str,
    model: str,
    *,
    delay: float,
    batch_size: int,
    concurrency: int,
) -> None:
    prefix = f"worker-{index + 1}"
    try:
        kept = asyncio.run(
            async_filter_records(
                records,
                api_key,
                model,
                delay=delay,
                batch_size=batch_size,
                concurrency=concurrency,
                progress_prefix=prefix,
            )
        )
    except Exception as exc:  # pragma: no cover - inter-process error propagation
        queue.put((index, None, str(exc)))
        return

    queue.put((index, kept, None))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Filter aging-theory reviews, retrieve full texts, and extract theory "
            "mentions in a single pass."
        )
    )
    parser.add_argument("--input", default="data/pipeline/start_reviews.json")
    parser.add_argument(
        "--output",
        default="data/pipeline/aging_theories.json",
        help=(
            "Path to the consolidated JSON payload containing enriched review "
            "records, raw theory snippets, and the canonical theory registry."
        ),
    )
    parser.add_argument(
        "--filtered-output",
        default="data/pipeline/filtered_reviews.json",
        help=(
            "Optional path storing the raw list of relevant reviews prior to "
            "full-text enrichment. Provide an empty string to skip writing this file."
        ),
    )
    parser.add_argument(
        "--failures",
        default=None,
        help=(
            "Optional path for logging failed PDF/full-text retrieval attempts. "
            "Defaults to <output>.failures.json when omitted."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help=(
            "OpenAI chat completion model identifier for the relevance filter. "
            "The default gpt-5-nano tier keeps this stage inside the ~$10 per "
            "million articles budget while preserving dependable batched "
            "decisions."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between OpenAI calls during filtering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Number of records to include in each LLM request. Larger batches reduce "
            "API calls but consume more tokens; pick a value that keeps prompts "
            "within the model's context window (e.g., 5-10 abstracts for gpt-5-nano)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent OpenAI requests during filtering.",
    )
    parser.add_argument(
        "--cache",
        help=(
            "Optional JSON cache storing previous LLM relevance decisions. When provided, "
            "the script skips cached records and appends new decisions after each batch."
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Number of OS processes used for filtering. When omitted the script "
            "auto-scales based on the record volume, aiming for roughly 25 reviews "
            "per worker while capping usage at the available CPU cores."
        ),
    )
    parser.add_argument(
        "--fulltext-processes",
        type=int,
        default=None,
        help=(
            "Worker count for the full-text enrichment stage. Defaults to a "
            "single process for ≤100 records and os.cpu_count() otherwise."
        ),
    )
    parser.add_argument(
        "--fulltext-concurrency",
        choices=("process", "thread"),
        default="process",
        help="Concurrency model for full-text retrieval workers (process/thread).",
    )
    parser.add_argument(
        "--entrez-interval",
        type=float,
        default=None,
        help=(
            "Minimum delay in seconds between Entrez requests. Defaults to "
            "PUBMED_RATE_INTERVAL or 0.34 when unset."
        ),
    )
    parser.add_argument(
        "--entrez-max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum retry attempts for Entrez HTTP 429/5xx responses. Defaults "
            "to PUBMED_MAX_ATTEMPTS or 5 when unset."
        ),
    )
    parser.add_argument(
        "--entrez-retry-wait",
        type=float,
        default=None,
        help=(
            "Initial wait time for Entrez retry backoff. Defaults to the "
            "effective Entrez interval or PUBMED_RETRY_WAIT when provided."
        ),
    )
    parser.add_argument(
        "--entrez-batch-size",
        type=int,
        default=None,
        help=(
            "Number of PMCIDs to request per Entrez efetch call. Defaults to "
            "PUBMED_BATCH_SIZE or 200 when unset."
        ),
    )
    parser.add_argument(
        "--extraction-model",
        default="gpt-5-nano",
        help=(
            "OpenAI model used for theory extraction and registry normalisation. "
            "Defaults to gpt-5-nano so long-form prompts remain affordable."
        ),
    )
    parser.add_argument(
        "--disambiguation-model",
        default="gpt-5-mini",
        help=(
            "OpenAI model used when reconciling extracted theory aliases against "
            "the canonical registry."
        ),
    )
    parser.add_argument(
        "--disambiguation-temperature",
        type=float,
        default=0.2,
        help=(
            "Sampling temperature applied to the disambiguation model when "
            "classifying theory synonyms."
        ),
    )
    parser.add_argument(
        "--extraction-delay",
        type=float,
        default=0.5,
        help="Seconds to wait between theory extraction OpenAI calls.",
    )
    parser.add_argument(
        "--extraction-concurrency",
        type=int,
        default=5,
        help=(
            "Maximum number of concurrent OpenAI requests per worker during the theory "
            "extraction stage."
        ),
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=12000,
        help="Maximum number of characters per review chunk sent to the extractor.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=1000,
        help="Characters of overlap between successive chunks during extraction.",
    )
    parser.add_argument(
        "--extraction-timeout",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT,
        help="Timeout (in seconds) for each theory extraction API request.",
    )
    parser.add_argument(
        "--extraction-processes",
        type=int,
        default=None,
        help=(
            "Number of worker processes for theory extraction. When omitted the "
            "script auto-scales based on the relevant record volume, defaulting "
            "to --processes when that flag is provided."
        ),
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    cache_path = args.cache
    cache_store = load_decision_cache(cache_path)
    if cache_path:
        print(
            f"Loaded {len(cache_store)} cached decisions from {cache_path}",
            flush=True,
        )

    failure_path = args.failures or f"{args.output}.failures.json"
    _existing_payload, existing_indexes = _load_existing_payload(args.output)
    existing_articles_by_identifier = existing_indexes["articles_by_identifier"]
    existing_articles_by_id = existing_indexes["articles_by_id"]
    existing_failures = _load_failure_log(failure_path)
    if existing_articles_by_identifier:
        print(
            f"Loaded {len(existing_articles_by_id)} previously enriched articles from {args.output}",
            flush=True,
        )

    total_records = len(records)
    if args.processes is not None and args.processes <= 0:
        print("--processes must be a positive integer", file=sys.stderr)
        return 2
    if args.fulltext_processes is not None and args.fulltext_processes <= 0:
        print("--fulltext-processes must be a positive integer", file=sys.stderr)
        return 3
    if args.extraction_processes is not None and args.extraction_processes <= 0:
        print("--extraction-processes must be a positive integer", file=sys.stderr)
        return 4
    if args.extraction_concurrency <= 0:
        print("--extraction-concurrency must be a positive integer", file=sys.stderr)
        return 4

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            print(
                f"Warning: invalid value for {name}={raw!r}; using {default:.2f}",
                file=sys.stderr,
            )
            return default

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            print(
                f"Warning: invalid value for {name}={raw!r}; using {default}",
                file=sys.stderr,
            )
            return default

    cpu_total = os.cpu_count() or 1
    min_records_per_worker = 25
    if total_records:
        auto_processes = min(cpu_total, max(1, math.ceil(total_records / min_records_per_worker)))
    else:
        auto_processes = 1
    processes = args.processes or auto_processes
    if total_records > 0:
        processes = min(processes, total_records)
    else:
        processes = 1

    if cache_path and processes > 1:
        print(
            "Cache persistence requires single-process execution; forcing --processes=1.",
            file=sys.stderr,
        )
        processes = 1

    if total_records == 0:
        kept: List[Dict] = []
    elif processes <= 1 or total_records <= 1:
        kept = asyncio.run(
            async_filter_records(
                records,
                api_key,
                args.model,
                delay=args.delay,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                cache=cache_store,
                cache_path=cache_path,
            )
        )
    else:
        queue: "Queue[Tuple[int, List[Dict] | None, str | None]]" = Queue()
        chunks = _split_records(list(records), processes)
        jobs = [
            Process(
                target=_worker_filter,
                args=(idx, chunk, queue, api_key, args.model),
                kwargs={
                    "delay": args.delay,
                    "batch_size": args.batch_size,
                    "concurrency": args.concurrency,
                },
            )
            for idx, chunk in enumerate(chunks)
            if chunk
        ]

        if not jobs:
            kept = []
        else:
            for job in jobs:
                job.start()

            results: List[Tuple[int, List[Dict]]] = []
            errors: List[Tuple[int, str]] = []
            for _ in range(len(jobs)):
                index, chunk_result, error = queue.get()
                if error:
                    errors.append((index, error))
                elif chunk_result is not None:
                    results.append((index, chunk_result))

            for job in jobs:
                job.join()

            if errors:
                for index, error in errors:
                    print(
                        f"worker-{index + 1} failed with error: {error}",
                        file=sys.stderr,
                    )
                return 5

            kept = [record for _, chunk in sorted(results, key=lambda item: item[0]) for record in chunk]

    filtered_output_path = (args.filtered_output or "").strip() or None
    if filtered_output_path:
        filtered_dir = os.path.dirname(filtered_output_path)
        if filtered_dir:
            os.makedirs(filtered_dir, exist_ok=True)
        with open(filtered_output_path, "w", encoding="utf-8") as fh:
            json.dump(kept, fh, ensure_ascii=False, indent=2)
        print(f"Saved {len(kept)} relevant reviews to {filtered_output_path}")
    else:
        print(f"Identified {len(kept)} relevant reviews")

    annotated_in_order: List[Optional[Dict]] = [None] * len(kept)
    new_records_for_enrichment: List[Dict] = []
    new_record_positions: List[int] = []
    reused_count = 0

    for idx, record in enumerate(kept):
        matched_article: Optional[Mapping[str, Any]] = None
        for key in _stable_identifier_keys(record):
            candidate = existing_articles_by_identifier.get(key)
            if candidate is not None:
                matched_article = candidate
                break
        if matched_article is None:
            article_id = record.get("article_id")
            if isinstance(article_id, str) and article_id.strip():
                matched_article = existing_articles_by_id.get(article_id.strip())
        if matched_article is None:
            new_records_for_enrichment.append(record)
            new_record_positions.append(idx)
            continue
        merged_article = dict(matched_article)
        for key, value in record.items():
            if key == "llm_filter":
                merged_article[key] = value
            elif key not in merged_article:
                merged_article[key] = value
        annotated_in_order[idx] = merged_article
        reused_count += 1

    enriched_new_records: List[Dict]
    failures: List[Dict]
    if not new_records_for_enrichment:
        enriched_new_records = []
        failures = []
    else:
        total_new = len(new_records_for_enrichment)
        if args.fulltext_processes is None:
            cpu_count = os.cpu_count() or 1
            fulltext_processes = cpu_count if total_new > 100 else 1
        else:
            fulltext_processes = args.fulltext_processes
        fulltext_processes = max(1, min(fulltext_processes, total_new))

        entrez_interval = (
            args.entrez_interval
            if args.entrez_interval is not None
            else _env_float("PUBMED_RATE_INTERVAL", 0.34)
        )
        entrez_max_attempts = (
            args.entrez_max_attempts
            if args.entrez_max_attempts is not None
            else _env_int("PUBMED_MAX_ATTEMPTS", 5)
        )
        raw_retry_wait = (
            args.entrez_retry_wait
            if args.entrez_retry_wait is not None
            else os.environ.get("PUBMED_RETRY_WAIT")
        )
        retry_wait_value = None
        if raw_retry_wait is not None:
            try:
                retry_wait_value = float(raw_retry_wait)
            except (TypeError, ValueError):
                print(
                    "Warning: invalid PUBMED_RETRY_WAIT value; using entrez interval",
                    file=sys.stderr,
                )
                retry_wait_value = None
        entrez_batch_size = (
            args.entrez_batch_size
            if args.entrez_batch_size is not None
            else _env_int("PUBMED_BATCH_SIZE", 200)
        )

        enriched_new_records, failures = enrich_records(
            new_records_for_enrichment,
            processes=fulltext_processes,
            concurrency=args.fulltext_concurrency,
            rate_limiter=None,
            entrez_interval=entrez_interval,
            entrez_max_attempts=entrez_max_attempts,
            entrez_retry_wait=retry_wait_value,
            entrez_batch_size=max(1, entrez_batch_size),
        )

    annotated_new_records: List[Dict]
    if enriched_new_records:
        relevant_count = len(enriched_new_records)
        if args.extraction_processes is None:
            if args.processes is not None:
                auto_extract = min(args.processes, cpu_total)
            else:
                auto_extract = min(
                    cpu_total,
                    max(1, math.ceil(relevant_count / min_records_per_worker)),
                )
        else:
            auto_extract = args.extraction_processes
        extraction_processes = max(1, min(auto_extract, relevant_count))

        annotated_new_records = extract_theories(
            enriched_new_records,
            api_key,
            args.extraction_model,
            args.extraction_delay,
            args.chunk_chars,
            args.chunk_overlap,
            extraction_processes,
            args.extraction_timeout,
            args.extraction_concurrency,
        )
    else:
        annotated_new_records = []

    for record, position in zip(annotated_new_records, new_record_positions):
        annotated_in_order[position] = record

    annotated_records = [record for record in annotated_in_order if record is not None]

    if annotated_records:
        theory_registry, synonym_registry = build_theory_registry(
            annotated_records,
            api_key,
            args.disambiguation_model,
            args.disambiguation_temperature,
            args.extraction_timeout,
        )
    else:
        theory_registry, synonym_registry = {}, {}

    aggregated_theories = sorted(theory_registry.keys()) if theory_registry else []
    raw_mentions = _collect_theory_mentions(annotated_records)
    fulltext_with_text = sum(
        1
        for record in annotated_records
        if isinstance(record.get("full_text"), str)
        and record.get("full_text", "").strip()
    )
    metadata = {
        "total_records": total_records,
        "relevant_records": len(annotated_records),
        "fulltext_with_content": fulltext_with_text,
        "theory_mentions": len(raw_mentions),
        "filter_model": args.model,
        "extraction_model": args.extraction_model,
        "disambiguation_model": args.disambiguation_model,
        "disambiguation_temperature": args.disambiguation_temperature,
        "reused_records": reused_count,
        "new_records_processed": len(new_records_for_enrichment),
    }

    combined_payload = {
        "articles": annotated_records,
        "raw_theory_mentions": raw_mentions,
        "aggregated_theories": aggregated_theories,
        "theory_registry": theory_registry,
        "synonym_registry": synonym_registry,
        "metadata": metadata,
    }

    combined_failures_log = _merge_failures(existing_failures, failures, annotated_records)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(combined_payload, fh, ensure_ascii=False, indent=2)

    failure_path = args.failures or f"{args.output}.failures.json"
    if combined_failures_log:
        failure_dir = os.path.dirname(failure_path)
        if failure_dir:
            os.makedirs(failure_dir, exist_ok=True)
        with open(failure_path, "w", encoding="utf-8") as fh:
            json.dump(combined_failures_log, fh, ensure_ascii=False, indent=2)
        print(f"Logged {len(combined_failures_log)} full-text failures to {failure_path}")
    elif args.failures and os.path.exists(failure_path):
        os.remove(failure_path)

    print(
        f"Saved {len(annotated_records)} enriched reviews and {len(raw_mentions)} theory snippets to {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

