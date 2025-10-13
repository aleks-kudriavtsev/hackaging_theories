"""Concurrency helpers for running classification and extraction in batches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Sequence, Tuple

from .extraction import QuestionAnswer, QuestionExtractor
from .literature import PaperMetadata
from .theories import TheoryAssignment, TheoryClassifier


def _batched_indices(total: int, batch_size: int) -> Iterable[List[int]]:
    step = max(1, batch_size)
    for start in range(0, total, step):
        end = min(total, start + step)
        yield list(range(start, end))


def classify_and_extract_parallel(
    papers: Sequence[PaperMetadata],
    classifier: TheoryClassifier,
    extractor: QuestionExtractor,
    *,
    workers: int | None = None,
    batch_size: int | None = None,
) -> Tuple[List[List[TheoryAssignment]], List[List[QuestionAnswer]]]:
    """Classify and extract answers for ``papers`` using optional worker threads."""

    if not papers:
        return [], []

    if batch_size is None:
        llm_client = getattr(classifier, "llm_client", None)
        if llm_client and getattr(llm_client, "config", None):
            batch_size = max(1, int(getattr(llm_client.config, "batch_size", 1)))
        else:
            batch_size = 16

    worker_count = 1
    if workers is not None:
        try:
            worker_count = int(workers)
        except (TypeError, ValueError):  # pragma: no cover - defensive programming
            worker_count = 1
    worker_count = max(1, worker_count)

    assignments_by_index: List[List[TheoryAssignment]] = [[] for _ in papers]
    answers_by_index: List[List[QuestionAnswer]] = [[] for _ in papers]

    def _process(indices: List[int]) -> Tuple[List[int], List[List[TheoryAssignment]], List[List[QuestionAnswer]]]:
        chunk_papers = [papers[i] for i in indices]
        chunk_assignments = classifier.classify_batch(chunk_papers)
        chunk_answers = [extractor.extract(paper) for paper in chunk_papers]
        return indices, chunk_assignments, chunk_answers

    index_batches = list(_batched_indices(len(papers), batch_size))

    if worker_count == 1:
        for indices in index_batches:
            idxs, chunk_assignments, chunk_answers = _process(indices)
            for position, assignment_list, answer_list in zip(idxs, chunk_assignments, chunk_answers):
                assignments_by_index[position] = assignment_list
                answers_by_index[position] = answer_list
        return assignments_by_index, answers_by_index

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_process, indices): indices for indices in index_batches}
        for future in as_completed(futures):
            idxs, chunk_assignments, chunk_answers = future.result()
            for position, assignment_list, answer_list in zip(idxs, chunk_assignments, chunk_answers):
                assignments_by_index[position] = assignment_list
                answers_by_index[position] = answer_list

    return assignments_by_index, answers_by_index


__all__ = ["classify_and_extract_parallel"]
