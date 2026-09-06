"""Scientific-boundary regression tests for the portable theory adapter."""

import copy
from pathlib import Path

import unittest
from tempfile import TemporaryDirectory

from theories_pipeline.biomarker_bridge import export_from_path, export_theory_pack

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6e45f01e5d108d07b11ceee79de850ddfc749446"


def make_pack(payload, input_file="registry.json"):
    return export_theory_pack(
        payload, source_repository="aleks-kudriavtsev/hackaging_theories",
        source_commit=COMMIT, registry_sha256="a" * 64, input_file=input_file,
    )


class TestBiomarkerBridge(unittest.TestCase):
    def test_repository_demonstration_cannot_be_scientific_evidence(self):
        for filename in ["aging_theories.json", "aging_ontology.json"]:
            self.check_fixture(filename)

    def check_fixture(self, filename):
        result = export_from_path(
            ROOT / "data/pipeline" / filename,
            source_repository="aleks-kudriavtsev/hackaging_theories", source_commit=COMMIT,
        )
        assert result["summary"] == {
            "total_theories": 3, "hypothesis_candidates": 0, "quarantined_theories": 3,
        }
        assert {n["theory_id"] for n in result["theories"]} == {
            "damage_accumulation", "telomere_attrition", "caloric_restriction",
        }
        assert all(n["scientific_validation"] == "not_assessed" for n in result["theories"])
        assert result["empirical_edges"] == result["theory_links"] == []


    def test_llm_validation_does_not_verify_a_real_looking_reference(self):
        payload = {"theory_registry": {"inflammaging": {
            "label": "Inflammaging", "doi": "10.1111/j.1749-6632.2000.tb06651.x",
            "confidence": 1.0, "validated": True, "theory_class": "mechanistic_hypothesis",
        }}}
        node = make_pack(payload)["theories"][0]
        assert node["status"] == "hypothesis_candidate"
        assert node["citations"][0]["verification_status"] == "identifier_unverified"
        assert node["citations"][0]["pmid"] is None
        assert node["scientific_validation"] == "not_assessed"


    def test_placeholder_id_is_quarantined_and_correction_is_explicit(self):
        source = {"theory_registry": {"t": {"label": "Theory", "supporting_articles": ["A1"]}}}
        assert make_pack(source)["theories"][0]["status"] == "quarantined_fixture"
        corrected = copy.deepcopy(source)
        corrected["articles"] = [{"id": "A1", "pmid": "10911963"}]
        node = make_pack(corrected)["theories"][0]
        assert node["status"] == "hypothesis_candidate"
        assert node["citations"][0]["pmid"] == "10911963"
        assert node["citations"][0]["verification_status"] == "identifier_unverified"
        assert source.get("articles") is None


    def test_fixture_path_and_explicit_fixture_metadata_are_quarantined(self):
        source = {"theory_registry": {"t": {"label": "Theory", "pmids": ["10911963"]}}}
        assert make_pack(source, "data/examples/seeds.json")["theories"][0]["status"] == "quarantined_fixture"
        source["is_synthetic"] = True
        assert make_pack(source)["theories"][0]["status"] == "quarantined_fixture"


    def test_conflicting_theory_ids_fail_instead_of_overwriting(self):
        source = {"ontology": {"final": {"groups": [{"theories": [
            {"theory_id": "t", "label": "Theory A"},
            {"theory_id": "t", "label": "Theory B"},
        ]}]}}}
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate theory ID"):
            make_pack(source)


    def test_duplicate_json_registry_key_fails(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "registry.json"
            source.write_text('{"theory_registry":{"t":{"label":"A"},"t":{"label":"B"}}}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                export_from_path(source, source_repository="owner/repo", source_commit=COMMIT)


    def test_aliases_do_not_merge_distinct_theories_or_infer_their_class(self):
        source = {"theory_registry": {
            "damage": {"label": "Damage accumulation", "aliases": ["Oxidative damage"]},
            "radical": {"label": "Free radical theory", "aliases": ["Oxidative damage"]},
        }}
        result = make_pack(source)
        assert result["summary"]["total_theories"] == 2
        assert all(node["theory_class"] == "unknown" for node in result["theories"])


    def test_numeric_local_article_id_is_not_guessed_as_pmid(self):
        source = {"theory_registry": {"t": {"label": "Theory", "supporting_articles": ["12345"]}}}
        assert make_pack(source)["theories"][0]["citations"][0]["pmid"] is None

if __name__ == "__main__":
    unittest.main()
