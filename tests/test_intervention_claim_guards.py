"""Regression checks against keyword-only longevity efficacy claims."""

import json
import unittest

from theories_pipeline.extraction import QuestionExtractor
from theories_pipeline.llm import LLMResponse


class DecliningLLM:
    def generate(self, messages_batch, *, model=None, temperature=None):
        return [LLMResponse(content=json.dumps({
            "answer": "unknown", "confidence": 0.9,
            "rationale": "The source does not establish efficacy.",
        }), cached=False)]


class TestInterventionClaimGuards(unittest.TestCase):
    def assert_screening_only(self, sentence, extractor=None):
        extractor = extractor or QuestionExtractor()
        answer, _, evidence_json, _, _ = extractor._classify("Q3", [sentence])
        self.assertEqual(answer, "Proposed longevity intervention")
        evidence = json.loads(evidence_json)
        self.assertEqual(evidence["heuristic"]["evidence"], sentence)
        self.assertEqual(evidence["heuristic"]["reason"], "requires source-level review")
        self.assertTrue(evidence["heuristic"]["screening_only"])
        self.assertFalse(evidence["heuristic"]["efficacy_assessed"])

    def test_negative_and_null_findings_preserve_negation(self):
        for sentence in [
            "The drug did not increase lifespan in mice.",
            "Treatment reduced survival by 20%.",
            "The intervention had no effect on longevity.",
        ]:
            with self.subTest(sentence=sentence):
                self.assert_screening_only(sentence)

    def test_untested_hypothesis_does_not_become_validated(self):
        self.assert_screening_only("This treatment may extend longevity but has not been tested.")

    def test_positive_quantitative_report_still_requires_source_review(self):
        for sentence in [
            "The drug increased lifespan by 25% in 80 mice (p<0.001).",
            "The dietary intervention is a validated longevity strategy.",
        ]:
            with self.subTest(sentence=sentence):
                self.assert_screening_only(sentence)

    def test_outcome_or_numbers_alone_do_not_imply_intervention(self):
        for sentence in [
            "Lifespan increased by 10% across the observed cohorts.",
            "Survival and longevity were recorded in 200 animals.",
            "Drugstores were counted in a study of longevity.",
        ]:
            with self.subTest(sentence=sentence):
                answer, _, evidence = QuestionExtractor()._classify_intervention([sentence])
                self.assertEqual(answer, "No intervention discussed")
                self.assertIsNone(evidence)

    def test_llm_abstention_keeps_safe_keyword_fallback(self):
        extractor = QuestionExtractor(llm_client=DecliningLLM())
        self.assert_screening_only("The drug did not increase lifespan in mice.", extractor)

    def test_custom_validation_keywords_cannot_restore_automatic_validation(self):
        extractor = QuestionExtractor({"label_keywords": {"Q3": {
            "Validated longevity intervention": ["miracle"],
            "Proposed longevity intervention": ["treatment"],
        }}})
        self.assert_screening_only("The treatment is a miracle for longevity.", extractor)


if __name__ == "__main__":
    unittest.main()
