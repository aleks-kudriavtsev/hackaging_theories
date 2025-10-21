"""Nine-question data extraction helpers for Hackaging."""

from __future__ import annotations

import json
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .literature import PaperMetadata
from .llm import LLMClient, LLMClientError, LLMMessage

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
    "Q1": (
        "Yes, quantitatively shown",
        "Yes, mentioned without data",
        "No evidence found",
    ),
    "Q2": (
        "Mechanism supported by experiments",
        "Mechanism hypothesized",
        "No mechanism discussed",
    ),
    "Q3": (
        "Validated longevity intervention",
        "Proposed longevity intervention",
        "No intervention discussed",
    ),
    "Q4": (
        "Changes appear irreversible",
        "Changes appear reversible",
        "Not discussed",
    ),
    "Q5": (
        "Yes, quantitatively shown",
        "Yes, mentioned without data",
        "No evidence found",
    ),
    "Q6": (
        "Primary focus of the paper",
        "Mentioned in passing",
        "Not mentioned",
    ),
    "Q7": (
        "Primary focus of the paper",
        "Mentioned in passing",
        "Not mentioned",
    ),
    "Q8": (
        "Link supported by data",
        "Link is speculative",
        "No link reported",
    ),
    "Q9": (
        "Experimental evidence presented",
        "Observational evidence presented",
        "No evidence presented",
    ),
}


QUESTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Q1": (
        "biomarker",
        "marker",
        "signature",
        "measured",
        "measurement",
        "assay",
        "quantified",
        "level",
        "concentration",
        "quantitative",
    ),
    "Q2": (
        "mechanism",
        "pathway",
        "molecular",
        "signaling",
        "process",
        "experiment",
        "demonstrate",
        "showed",
        "knockout",
        "mutation",
        "assay",
        "inhibited",
    ),
    "Q3": (
        "intervention",
        "treatment",
        "therapy",
        "drug",
        "compound",
        "supplement",
        "regimen",
        "lifespan",
        "longevity",
        "validated",
        "proposed",
    ),
    "Q4": (
        "irreversible",
        "reversible",
        "permanent",
        "restored",
        "rescued",
        "recovered",
    ),
    "Q5": (
        "cross-species",
        "comparative",
        "predict",
        "predictor",
        "phylogenetic",
        "dataset",
        "regression",
        "correlation",
    ),
    "Q6": ("naked mole rat", "heterocephalus glaber", "nmr", "longevity"),
    "Q7": ("avian", "bird", "avian longevity", "passerine", "galliform"),
    "Q8": (
        "body size",
        "mass",
        "weight",
        "allometry",
        "scaling",
        "longevity",
        "correlated",
    ),
    "Q9": (
        "calorie restriction",
        "caloric restriction",
        "dietary restriction",
        "restricted diet",
        "trial",
        "experiment",
    ),
}


QUESTION_LOOKUP: Dict[str, str] = {identifier: question for identifier, question in QUESTIONS}

DEFAULT_MAX_EVIDENCE_SENTENCES = 5


@dataclass(frozen=True)
class _LLMResult:
    answer: Optional[str]
    confidence: Optional[float]
    rationale: Optional[str]
    raw: str
    candidate_sentences: Tuple[str, ...]


@dataclass(frozen=True)
class QuestionAnswer:
    paper_id: str
    question_id: str
    question: str
    answer: str
    confidence: float
    evidence: Optional[str] = None
    heuristic_confidence: Optional[float] = None
    gpt_confidence: Optional[float] = None


class QuestionExtractor:
    """Rule-based extractor for the nine Hackaging challenge questions."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        llm_client: Optional[LLMClient] = None,
    ):
        config = config or {}
        overrides = config.get("label_keywords", {})
        self.label_keywords: Dict[str, Dict[str, Sequence[str]]] = {
            question_id: {
                label: tuple(keyword.lower() for keyword in keywords)
                for label, keywords in labels.items()
            }
            for question_id, labels in overrides.items()
        }
        self.llm_client = llm_client

        llm_config = config.get("llm", {}) if isinstance(config, Mapping) else {}
        self.llm_enabled = bool(llm_config.get("enabled", True))
        self.llm_max_sentences = int(
            llm_config.get("max_sentences", DEFAULT_MAX_EVIDENCE_SENTENCES)
        )
        keyword_overrides = llm_config.get("keywords", {}) if isinstance(llm_config, Mapping) else {}
        self.llm_keywords: Dict[str, Tuple[str, ...]] = {
            question_id: tuple(
                {
                    keyword.lower()
                    for keyword in (
                        *(QUESTION_KEYWORDS.get(question_id, ())),
                        *tuple(keyword_overrides.get(question_id, ())),
                    )
                }
            )
            for question_id in QUESTION_CHOICES
        }
        self.llm_system_prompt = llm_config.get(
            "system_prompt",
            (
                "You are a careful scientific evidence analyst. "
                "Choose the best categorical answer for each question using only the provided sentences. "
                "Respond with a JSON object containing keys 'answer', 'confidence', and 'rationale'. "
                "The 'answer' must exactly match one of the allowed answers. "
                "Use 'unknown' when the evidence is insufficient."
            ),
        )
        self.llm_request_template = llm_config.get(
            "request_template",
            (
                "Question: {question}\n"
                "Allowed answers: {choices}\n"
                "Evidence sentences:\n{evidence}\n"
                "Respond strictly in JSON with keys 'answer', 'confidence', and 'rationale'. "
                "The 'answer' must exactly match one of the allowed answers."
            ),
        )
        decline_tokens = llm_config.get("decline_answers", ("unknown", "none", "no_answer"))
        self.llm_decline_answers = {
            str(token).strip().lower() for token in decline_tokens if str(token).strip()
        }

        calibration_cfg = config.get("calibration", {}) if isinstance(config, Mapping) else {}
        self.gpt_weight = float(calibration_cfg.get("gpt_weight", 0.6))
        self.gpt_override_threshold = float(
            calibration_cfg.get("override_threshold", 0.85)
        )
        self.agreement_boost = float(calibration_cfg.get("agreement_boost", 0.1))
        self.disagreement_penalty = float(
            calibration_cfg.get("disagreement_penalty", 0.1)
        )

    def extract(self, paper: PaperMetadata) -> List[QuestionAnswer]:
        analysis_text = paper.analysis_text if paper.analysis_text else paper.abstract
        text = " ".join(part for part in (paper.title, analysis_text) if part)
        answers: List[QuestionAnswer] = []
        sentences = _split_sentences(text)
        for question_id, question in QUESTIONS:
            answer, confidence, evidence, heuristic_conf, gpt_conf = self._classify(
                question_id, sentences
            )
            answers.append(
                QuestionAnswer(
                    paper.identifier,
                    question_id,
                    question,
                    answer,
                    confidence,
                    evidence,
                    heuristic_confidence=heuristic_conf,
                    gpt_confidence=gpt_conf,
                )
            )
        return answers

    def _classify(
        self, question_id: str, sentences: Sequence[str]
    ) -> Tuple[str, float, Optional[str], float, Optional[float]]:
        heuristic_answer: str
        heuristic_conf: float
        heuristic_evidence: Optional[str]

        if question_id == "Q1":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_biomarker(
                sentences
            )
        elif question_id == "Q2":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_mechanism(
                sentences
            )
        elif question_id == "Q3":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_intervention(
                sentences
            )
        elif question_id == "Q4":
            heuristic_answer, heuristic_conf, heuristic_evidence = (
                self._classify_irreversibility(sentences)
            )
        elif question_id == "Q5":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_cross_species(
                sentences
            )
        elif question_id == "Q6":
            heuristic_answer, heuristic_conf, heuristic_evidence = (
                self._classify_naked_mole_rat(sentences)
            )
        elif question_id == "Q7":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_avian(
                sentences
            )
        elif question_id == "Q8":
            heuristic_answer, heuristic_conf, heuristic_evidence = self._classify_body_size(
                sentences
            )
        elif question_id == "Q9":
            heuristic_answer, heuristic_conf, heuristic_evidence = (
                self._classify_calorie_restriction(sentences)
            )
        else:
            fallback_choices = QUESTION_CHOICES.get(question_id)
            heuristic_answer = (
                fallback_choices[-1]
                if fallback_choices
                else "No evidence found"
            )
            heuristic_conf, heuristic_evidence = 0.0, None

        gpt_result = None
        if self.llm_client and self.llm_enabled:
            gpt_result = self._invoke_llm(question_id, sentences, heuristic_evidence)

        final_answer, final_confidence, evidence, gpt_confidence = self._combine_results(
            question_id,
            heuristic_answer,
            heuristic_conf,
            heuristic_evidence,
            gpt_result,
        )
        return final_answer, final_confidence, evidence, heuristic_conf, gpt_confidence

    def _classify_biomarker(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        quantitative_label, qualitative_label, none_label = QUESTION_CHOICES["Q1"]
        keywords = self._get_keywords(
            "Q1", qualitative_label, ("biomarker", "marker", "signature")
        )
        quant_terms = self._get_keywords(
            "Q1",
            quantitative_label,
            ("measured", "level", "concentration", "quantified", "assay", "mg", "ng", "pg"),
        )
        digit_pattern = re.compile(r"\d")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                if any(term in lowered for term in quant_terms) or digit_pattern.search(sentence):
                    return quantitative_label, 0.9, sentence.strip()
                return qualitative_label, 0.6, sentence.strip()
        return none_label, 0.1, None

    def _classify_mechanism(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        supported_label, hypothesis_label, none_label = QUESTION_CHOICES["Q2"]
        mechanism_terms = self._get_keywords(
            "Q2",
            hypothesis_label,
            ("mechanism", "pathway", "molecular", "signaling", "process"),
        )
        evidence_terms = self._get_keywords(
            "Q2",
            supported_label,
            ("experiment", "demonstrate", "showed", "knockout", "mutation", "assay", "inhibited"),
        )
        hypothesis_terms = ("suggest", "propose", "hypothesize", "may", "could")
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in mechanism_terms):
                if any(term in lowered for term in evidence_terms):
                    return supported_label, 0.85, sentence.strip()
                if any(term in lowered for term in hypothesis_terms):
                    return hypothesis_label, 0.6, sentence.strip()
                return hypothesis_label, 0.55, sentence.strip()
        return none_label, 0.1, None

    def _classify_intervention(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        validated_label, proposed_label, none_label = QUESTION_CHOICES["Q3"]
        intervention_terms = self._get_keywords(
            "Q3",
            proposed_label,
            ("intervention", "treatment", "therapy", "drug", "compound", "supplement", "regimen"),
        )
        validation_terms = self._get_keywords(
            "Q3",
            validated_label,
            ("extends lifespan", "increased lifespan", "longevity", "survival", "lifespan"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in intervention_terms):
                if any(term in lowered for term in validation_terms):
                    return validated_label, 0.9, sentence.strip()
                speculative = ("potential", "candidate", "may", "could", "propose", "suggest")
                if any(term in lowered for term in speculative):
                    return proposed_label, 0.6, sentence.strip()
                if "lifespan" in lowered or "longevity" in lowered:
                    return validated_label, 0.75, sentence.strip()
                return proposed_label, 0.55, sentence.strip()
        return none_label, 0.1, None

    def _classify_irreversibility(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        irreversible_label, reversible_label, none_label = QUESTION_CHOICES["Q4"]
        irreversible_terms = self._get_keywords(
            "Q4",
            irreversible_label,
            ("irreversible", "permanent", "irreversibly", "cannot be reversed"),
        )
        reversible_terms = self._get_keywords(
            "Q4",
            reversible_label,
            ("reversible", "reversibly", "restored", "rescued", "recovered"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in irreversible_terms):
                return irreversible_label, 0.8, sentence.strip()
            if any(term in lowered for term in reversible_terms):
                return reversible_label, 0.75, sentence.strip()
        return none_label, 0.2, None

    def _classify_cross_species(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        quantitative_label, qualitative_label, none_label = QUESTION_CHOICES["Q5"]
        predictor_terms = self._get_keywords(
            "Q5",
            qualitative_label,
            ("cross-species", "comparative", "predict", "predictor", "phylogenetic", "across species"),
        )
        quant_terms = self._get_keywords(
            "Q5",
            quantitative_label,
            ("model", "regression", "correlation", "estimate", "coefficient", "dataset", "analysis"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in predictor_terms):
                if any(term in lowered for term in quant_terms) or re.search(r"\d", sentence):
                    return quantitative_label, 0.85, sentence.strip()
                return qualitative_label, 0.55, sentence.strip()
        return none_label, 0.1, None

    def _classify_naked_mole_rat(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        primary_label, mention_label, none_label = QUESTION_CHOICES["Q6"]
        focus_terms = self._get_keywords(
            "Q6",
            primary_label,
            ("naked mole rat", "heterocephalus glaber", "nmr"),
        )
        for sentence in sentences:
            lowered = sentence.lower()
            if any(term in lowered for term in focus_terms):
                if "study" in lowered or "experiment" in lowered or "analysis" in lowered:
                    return primary_label, 0.85, sentence.strip()
                return mention_label, 0.6, sentence.strip()
        return none_label, 0.1, None

    def _classify_avian(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        primary_label, mention_label, none_label = QUESTION_CHOICES["Q7"]
        avian_terms = self._get_keywords(
            "Q7",
            primary_label,
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
                        return primary_label, 0.8, sentence.strip()
                    return mention_label, 0.55, sentence.strip()
                return mention_label, 0.5, sentence.strip()
        return none_label, 0.1, None

    def _classify_body_size(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        supported_label, speculative_label, none_label = QUESTION_CHOICES["Q8"]
        body_size_terms = self._get_keywords(
            "Q8",
            supported_label,
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
                    return supported_label, 0.8, sentence.strip()
                return speculative_label, 0.55, sentence.strip()
        return none_label, 0.1, None

    def _classify_calorie_restriction(self, sentences: Sequence[str]) -> Tuple[str, float, Optional[str]]:
        experimental_label, observational_label, none_label = QUESTION_CHOICES["Q9"]
        restriction_terms = self._get_keywords(
            "Q9",
            experimental_label,
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
                    return experimental_label, 0.85, sentence.strip()
                if any(term in lowered for term in observational_terms):
                    return observational_label, 0.6, sentence.strip()
                return observational_label, 0.55, sentence.strip()
        return none_label, 0.1, None

    def _candidate_sentences(
        self,
        question_id: str,
        sentences: Sequence[str],
        heuristic_evidence: Optional[str],
    ) -> Tuple[str, ...]:
        keywords = self.llm_keywords.get(question_id, ())
        lowered_keywords = tuple(keyword.lower() for keyword in keywords)
        collected: List[str] = []
        seen: set[str] = set()

        def _add(sentence: str) -> None:
            normalized = sentence.strip()
            if normalized and normalized not in seen:
                collected.append(normalized)
                seen.add(normalized)

        if heuristic_evidence:
            _add(heuristic_evidence)

        for sentence in sentences:
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in lowered_keywords):
                _add(sentence)
            if len(collected) >= self.llm_max_sentences:
                break

        if not collected:
            for sentence in sentences[: self.llm_max_sentences]:
                _add(sentence)

        return tuple(collected[: self.llm_max_sentences])

    def _invoke_llm(
        self,
        question_id: str,
        sentences: Sequence[str],
        heuristic_evidence: Optional[str],
    ) -> Optional[_LLMResult]:
        if not self.llm_client:
            return None

        candidate_sentences = self._candidate_sentences(
            question_id, sentences, heuristic_evidence
        )
        evidence_block = "\n".join(f"- {sentence}" for sentence in candidate_sentences)
        if not evidence_block:
            evidence_block = "(no candidate evidence sentences were found)"

        prompt = self.llm_request_template.format(
            question=QUESTION_LOOKUP.get(question_id, question_id),
            choices=", ".join(QUESTION_CHOICES.get(question_id, ())),
            evidence=evidence_block,
        )
        messages = [
            LLMMessage(role="system", content=self.llm_system_prompt),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = self.llm_client.generate([messages])[0]
        except (LLMClientError, IndexError):  # pragma: no cover - defensive fallback
            return None

        raw_content = response.content.strip()
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            return _LLMResult(None, None, None, raw_content, candidate_sentences)

        answer_payload = payload.get("answer")
        confidence_value: Optional[float]
        confidence_raw = payload.get("confidence")

        answer: Optional[str]
        if isinstance(answer_payload, Mapping):
            if confidence_raw is None:
                confidence_raw = answer_payload.get("confidence")
            answer = (
                answer_payload.get("label")
                or answer_payload.get("value")
                or answer_payload.get("answer")
                or answer_payload.get("text")
            )
        elif isinstance(answer_payload, Sequence) and not isinstance(answer_payload, (str, bytes)):
            answer = None
            for entry in answer_payload:
                if isinstance(entry, Mapping):
                    if confidence_raw is None:
                        confidence_raw = entry.get("confidence")
                    answer = (
                        entry.get("label")
                        or entry.get("value")
                        or entry.get("answer")
                        or entry.get("text")
                    )
                elif isinstance(entry, str):
                    answer = entry
                if answer:
                    break
        else:
            answer = answer_payload

        try:
            confidence_value = float(confidence_raw)
        except (TypeError, ValueError):
            confidence_value = None
        if confidence_value is not None:
            confidence_value = max(0.0, min(1.0, confidence_value))
        rationale = payload.get("rationale")
        if isinstance(rationale, list):
            rationale = " ".join(str(part) for part in rationale)
        elif rationale is not None:
            rationale = str(rationale)

        if isinstance(answer, str):
            answer = answer.strip()
        else:
            answer = None

        return _LLMResult(answer, confidence_value, rationale, raw_content, candidate_sentences)

    def _combine_results(
        self,
        question_id: str,
        heuristic_answer: str,
        heuristic_confidence: float,
        heuristic_evidence: Optional[str],
        gpt_result: Optional[_LLMResult],
    ) -> Tuple[str, float, str, Optional[float]]:
        allowed_answers = QUESTION_CHOICES.get(question_id, ())
        allowed_lookup = {choice.lower(): choice for choice in allowed_answers}
        final_answer = heuristic_answer
        final_confidence = heuristic_confidence
        gpt_confidence: Optional[float] = None

        evidence_payload: Dict[str, Any] = {
            "heuristic": {
                "answer": heuristic_answer,
                "confidence": heuristic_confidence,
                "evidence": heuristic_evidence,
            }
        }

        if gpt_result is not None:
            normalized_answer = (gpt_result.answer or "").strip()
            lowered = normalized_answer.lower()
            gpt_confidence = gpt_result.confidence
            canonical_answer = allowed_lookup.get(lowered)
            decline_choice = lowered in self.llm_decline_answers
            if canonical_answer:
                normalized_answer = canonical_answer
            valid_choice = canonical_answer is not None and not decline_choice

            evidence_payload["gpt"] = {
                "answer": gpt_result.answer,
                "confidence": gpt_confidence,
                "rationale": gpt_result.rationale,
                "raw": gpt_result.raw,
                "candidate_sentences": list(gpt_result.candidate_sentences),
            }

            if valid_choice:
                weight = min(max(self.gpt_weight, 0.0), 1.0)
                gpt_conf_value = gpt_confidence if gpt_confidence is not None else 0.0
                if normalized_answer == heuristic_answer:
                    agreement_boost = self.agreement_boost * min(
                        heuristic_confidence, gpt_conf_value
                    )
                    final_confidence = min(
                        1.0, max(heuristic_confidence, gpt_conf_value) + agreement_boost
                    )
                    final_answer = normalized_answer
                else:
                    if (
                        gpt_conf_value >= self.gpt_override_threshold
                        and gpt_conf_value >= heuristic_confidence
                    ):
                        final_answer = normalized_answer
                        final_confidence = min(1.0, gpt_conf_value + self.agreement_boost / 2)
                    else:
                        combined = (1 - weight) * heuristic_confidence + weight * gpt_conf_value
                        final_confidence = max(0.0, min(1.0, combined))
                        if gpt_conf_value > heuristic_confidence:
                            final_answer = normalized_answer
                gpt_confidence = gpt_conf_value
            else:
                # Decline or invalid choice – favour heuristic answer but capture rationale
                if not decline_choice and normalized_answer:
                    final_confidence = max(0.0, heuristic_confidence - self.disagreement_penalty)
        else:
            weight = min(max(self.gpt_weight, 0.0), 1.0)
            final_confidence = max(
                0.0,
                min(1.0, heuristic_confidence * (1 - weight) + weight * heuristic_confidence),
            )

        evidence = json.dumps(evidence_payload, ensure_ascii=False)
        return final_answer, max(0.0, min(1.0, final_confidence)), evidence, gpt_confidence

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
