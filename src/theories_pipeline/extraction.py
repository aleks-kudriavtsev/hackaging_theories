"""Nine-question data extraction helpers for Hackaging."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .literature import PaperMetadata

QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("Q1", "Does the paper report a biomarker of aging?"),
    ("Q2", "Does the paper describe a molecular mechanism for longevity?"),
    ("Q3", "Does the study test or propose a longevity intervention?"),
    ("Q4", "Does the paper address irreversibility or reversibility of aging changes?"),
    ("Q5", "Does the paper provide cross-species lifespan predictors?"),
    ("Q6", "Does the paper focus on naked mole rat longevity?"),
    ("Q7", "Does the paper focus on avian longevity?"),
    ("Q8", "Does the paper link body size to longevity?"),
    ("Q9", "Does the paper examine calorie restriction and lifespan?"),
)


QUESTION_CHOICES: Dict[str, Tuple[str, ...]] = {
    "Q1": ("yes_quantitative", "yes_unverified", "no"),
    "Q2": ("mechanistic_evidence", "mechanistic_hypothesis", "no"),
    "Q3": ("validated_intervention", "proposed_intervention", "no"),
    "Q4": ("irreversible", "reversible", "not_discussed"),
    "Q5": ("yes_quantitative", "yes_unverified", "no"),
    "Q6": ("primary_focus", "mentioned", "no"),
    "Q7": ("primary_focus", "mentioned", "no"),
    "Q8": ("supported", "speculative", "no"),
    "Q9": ("experimental", "observational", "no"),
}


@dataclass(frozen=True)
class QuestionAnswer:
    paper_id: str
    question_id: str
    question: str
    answer: str
    confidence: float
    evidence: Optional[str] = None


class QuestionExtractor:
    """Rule-based extractor for the nine Hackaging challenge questions."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        config = config or {}
        overrides = config.get("label_keywords", {})
        self.label_keywords: Dict[str, Dict[str, Sequence[str]]] = {
            question_id: {
                label: tuple(keyword.lower() for keyword in keywords)
                for label, keywords in labels.items()
            }
            for question_id, labels in overrides.items()
        }

    def extract(self, paper: PaperMetadata) -> List[QuestionAnswer]:
        text = " ".join([paper.title, paper.abstract])
        answers: List[QuestionAnswer] = []
        sentences = _split_sentences(text)
        for question_id, question in QUESTIONS:
            answer, confidence, evidence = self._classify(question_id, sentences)
            answers.append(
                QuestionAnswer(
                    paper.identifier,
                    question_id,
                    question,
                    answer,
                    confidence,
                    evidence,
                )
            )
        return answers

    def _classify(self, question_id: str, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        if question_id == "Q1":
            return self._classify_biomarker(sentences)
        if question_id == "Q2":
            return self._classify_mechanism(sentences)
        if question_id == "Q3":
            return self._classify_intervention(sentences)
        if question_id == "Q4":
            return self._classify_irreversibility(sentences)
        if question_id == "Q5":
            return self._classify_cross_species(sentences)
        if question_id == "Q6":
            return self._classify_naked_mole_rat(sentences)
        if question_id == "Q7":
            return self._classify_avian(sentences)
        if question_id == "Q8":
            return self._classify_body_size(sentences)
        if question_id == "Q9":
            return self._classify_calorie_restriction(sentences)
        return "no", 0.0, None

    def _classify_biomarker(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        keywords = self._get_keywords("Q1", "yes_unverified", ("biomarker", "marker", "signature"))
        quant_terms = self._get_keywords(
            "Q1",
            "yes_quantitative",
            ("measured", "level", "concentration", "quantified", "assay", "mg", "ng", "pg"),
        )
        digit_pattern = re.compile(r"\d")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                if any(term in lowered for term in quant_terms) or digit_pattern.search(sentence):
                    return "yes_quantitative", 0.9, sentence.strip()
                return "yes_unverified", 0.6, sentence.strip()
        return "no", 0.1, None

    def _classify_mechanism(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        mechanism_terms = self._get_keywords(
            "Q2",
            "mechanistic_hypothesis",
            ("mechanism", "pathway", "molecular", "signaling", "process"),
        )
        evidence_terms = self._get_keywords(
            "Q2",
            "mechanistic_evidence",
            ("experiment", "demonstrate", "showed", "knockout", "mutation", "assay", "inhibited"),
        )
        hypothesis_terms = ("suggest", "propose", "hypothesize", "may", "could")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in mechanism_terms):
                if any(term in lowered for term in evidence_terms):
                    return "mechanistic_evidence", 0.85, sentence.strip()
                if any(term in lowered for term in hypothesis_terms):
                    return "mechanistic_hypothesis", 0.6, sentence.strip()
                return "mechanistic_hypothesis", 0.55, sentence.strip()
        return "no", 0.1, None

    def _classify_intervention(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        intervention_terms = self._get_keywords(
            "Q3",
            "proposed_intervention",
            ("intervention", "treatment", "therapy", "drug", "compound", "supplement", "regimen"),
        )
        validation_terms = self._get_keywords(
            "Q3",
            "validated_intervention",
            ("extends lifespan", "increased lifespan", "longevity", "survival", "lifespan"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in intervention_terms):
                if any(term in lowered for term in validation_terms):
                    return "validated_intervention", 0.9, sentence.strip()
                speculative = ("potential", "candidate", "may", "could", "propose", "suggest")
                if any(term in lowered for term in speculative):
                    return "proposed_intervention", 0.6, sentence.strip()
                if "lifespan" in lowered or "longevity" in lowered:
                    return "validated_intervention", 0.75, sentence.strip()
                return "proposed_intervention", 0.55, sentence.strip()
        return "no", 0.1, None

    def _classify_irreversibility(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        irreversible_terms = self._get_keywords(
            "Q4", "irreversible", ("irreversible", "permanent", "irreversibly", "cannot be reversed")
        )
        reversible_terms = self._get_keywords(
            "Q4", "reversible", ("reversible", "reversibly", "restored", "rescued", "recovered")
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in irreversible_terms):
                return "irreversible", 0.8, sentence.strip()
            if any(term in lowered for term in reversible_terms):
                return "reversible", 0.75, sentence.strip()
        return "not_discussed", 0.2, None

    def _classify_cross_species(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        predictor_terms = self._get_keywords(
            "Q5",
            "yes_unverified",
            ("cross-species", "comparative", "predict", "predictor", "phylogenetic", "across species"),
        )
        quant_terms = self._get_keywords(
            "Q5",
            "yes_quantitative",
            ("model", "regression", "correlation", "estimate", "coefficient", "dataset", "analysis"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in predictor_terms):
                if any(term in lowered for term in quant_terms) or re.search(r"\d", sentence):
                    return "yes_quantitative", 0.85, sentence.strip()
                return "yes_unverified", 0.55, sentence.strip()
        return "no", 0.1, None

    def _classify_naked_mole_rat(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        focus_terms = self._get_keywords(
            "Q6",
            "primary_focus",
            ("naked mole rat", "heterocephalus glaber", "nmr"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in focus_terms):
                if "study" in lowered or "experiment" in lowered or "analysis" in lowered:
                    return "primary_focus", 0.85, sentence.strip()
                return "mentioned", 0.6, sentence.strip()
        return "no", 0.1, None

    def _classify_avian(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        avian_terms = self._get_keywords(
            "Q7",
            "primary_focus",
            ("avian", "bird", "passerine", "avian longevity", "galliform"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in avian_terms):
                if "longevity" in lowered or "lifespan" in lowered:
                    if (
                        "study" in lowered
                        or "analysis" in lowered
                        or "analyz" in lowered
                    ):
                        return "primary_focus", 0.8, sentence.strip()
                    return "mentioned", 0.55, sentence.strip()
                return "mentioned", 0.5, sentence.strip()
        return "no", 0.1, None

    def _classify_body_size(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        body_size_terms = self._get_keywords(
            "Q8",
            "supported",
            ("body size", "mass", "weight", "allometry", "scaling"),
        )
        longevity_terms = ("longevity", "lifespan", "life span", "survival")
        quantitative_terms = ("correlated", "association", "regression", "model", "predict")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in body_size_terms) and any(
                term in lowered for term in longevity_terms
            ):
                if any(term in lowered for term in quantitative_terms) or re.search(r"\d", sentence):
                    return "supported", 0.8, sentence.strip()
                return "speculative", 0.55, sentence.strip()
        return "no", 0.1, None

    def _classify_calorie_restriction(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        restriction_terms = self._get_keywords(
            "Q9",
            "experimental",
            (
                "calorie restriction",
                "caloric restriction",
                "dietary restriction",
                "restricted diet",
                "calorie-restricted",
            ),
        )
        observational_terms = ("observational", "survey", "cohort", "epidemiological")
        experimental_terms = ("experiment", "trial", "controlled", "randomized", "intervention")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in restriction_terms):
                if any(term in lowered for term in experimental_terms):
                    return "experimental", 0.85, sentence.strip()
                if any(term in lowered for term in observational_terms):
                    return "observational", 0.6, sentence.strip()
                return "observational", 0.55, sentence.strip()
        return "no", 0.1, None

    def _get_keywords(
        self,
        question_id: str,
        label: str,
        defaults: Sequence[str],
    ) -> Sequence[str]:
        return self.label_keywords.get(question_id, {}).get(label, defaults)


def _split_sentences(text: str) -> List[str]:
    cleaned = text.replace("\n", " ")
    parts = [part.strip() for part in cleaned.split(".")]
    return [part for part in parts if part]
