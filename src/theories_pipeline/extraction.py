"""Nine-question data extraction helpers for Hackaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from .literature import PaperMetadata

QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("Q1", "What aging theory is the paper engaging with?"),
    ("Q2", "What research question is posed?"),
    ("Q3", "What study design or method is used?"),
    ("Q4", "What population or sample is examined?"),
    ("Q5", "What outcomes are measured?"),
    ("Q6", "What interventions or exposures are considered?"),
    ("Q7", "What are the main findings?"),
    ("Q8", "What limitations are discussed?"),
    ("Q9", "What future directions are suggested?"),
)


@dataclass(frozen=True)
class QuestionAnswer:
    paper_id: str
    question_id: str
    question: str
    answer: str


class QuestionExtractor:
    """Keyword driven extractor for the nine standard Hackaging questions."""

    def __init__(self, keyword_templates: Mapping[str, Iterable[str]] | None = None):
        if keyword_templates is None:
            keyword_templates = {
                "Q1": ["theory", "framework"],
                "Q2": ["research question", "aim"],
                "Q3": ["method", "design"],
                "Q4": ["participants", "population"],
                "Q5": ["outcome", "measure"],
                "Q6": ["intervention", "exposure"],
                "Q7": ["finding", "result"],
                "Q8": ["limitation", "constraint"],
                "Q9": ["future", "further research"],
            }
        self.keyword_templates: Dict[str, List[str]] = {
            key: [kw.lower() for kw in values]
            for key, values in keyword_templates.items()
        }

    def extract(self, paper: PaperMetadata) -> List[QuestionAnswer]:
        text = " ".join([paper.title, paper.abstract])
        answers: List[QuestionAnswer] = []
        for question_id, question in QUESTIONS:
            answer = self._derive_answer(question_id, question, text)
            answers.append(QuestionAnswer(paper.identifier, question_id, question, answer))
        return answers

    def _derive_answer(
        self,
        question_id: str,
        question: str,
        text: str,
    ) -> str:
        keywords = self.keyword_templates.get(question_id, [])
        if not keywords:
            return _fallback_answer(question, text)
        sentences = _split_sentences(text)
        for sentence in sentences:
            lower_sentence = sentence.lower()
            multi_keywords = [keyword for keyword in keywords if " " in keyword]
            if multi_keywords and all(keyword in lower_sentence for keyword in multi_keywords):
                return sentence.strip()
            if any(keyword in lower_sentence for keyword in keywords):
                return sentence.strip()
        return _fallback_answer(question, text)


def _fallback_answer(question: str, text: str) -> str:
    snippet = text.strip().split(". ")[0].strip()
    snippet = snippet or "Information not available"
    return f"{question} - {snippet}"


def _split_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ")
    parts = [part.strip() for part in cleaned.split(".")]
    return [part for part in parts if part]
