"""Contract and scientific-boundary checks for real curated discovery data."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from theories_pipeline.biomarker_bridge import export_from_path, export_theory_pack
from theories_pipeline.curated_catalog import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/curated/aging_theory_catalog.json"
COMMIT = "cf3857e14f3fbd1c6093cdfed7b41f2e78033558"


def export(payload):
    return export_theory_pack(payload, source_repository="aleks-kudriavtsev/hackaging_theories",
                              source_commit=COMMIT, registry_sha256="a" * 64,
                              input_file="data/curated/aging_theory_catalog.json")


class TestCuratedCatalog(unittest.TestCase):
    def setUp(self):
        self.source = json.loads(CATALOG.read_text())

    def test_real_catalog_is_discovery_ready_without_empirical_promotion(self):
        result = export(self.source)
        self.assertEqual(result["summary"], {"total_theories": 6, "hypothesis_candidates": 6,
                                            "quarantined_theories": 0})
        self.assertEqual(result["empirical_edges"], [])
        self.assertEqual(result["theory_links"], [])
        consumer = result["consumer_config"]
        self.assertEqual(consumer, self.source["consumer_config"])
        self.assertEqual(result["provenance"]["consumer_config_sha256"], canonical_sha256(consumer))
        self.assertEqual(len(consumer["predictions"]), 7)
        self.assertTrue(all(not p["confirmed"] and not p["observation_ids"] for p in consumer["predictions"]))
        self.assertTrue(all(t["scientific_validation"] == "not_assessed" for t in consumer["theories"]))
        self.assertTrue(all(c["verification_status"] == "identifier_unverified"
                            for t in result["theories"] for c in t["citations"]))

    def test_every_concept_has_all_three_context_levels_and_discriminating_questions(self):
        consumer = export(self.source)["consumer_config"]
        nodes = {n["id"]: n for n in consumer["nodes"]}
        for theory in consumer["theories"]:
            tid = theory["theory_id"]
            self.assertEqual({nodes[l["node_id"]]["level"] for l in consumer["links"] if l["theory_id"] == tid}, {"L1", "L2", "L3"})
            self.assertTrue(any(tid in p["theory_ids"] and p["falsification_criterion"] for p in consumer["predictions"]))
        telomere = next(p for p in consumer["predictions"] if p["id"] == "P_TELOMERE_INCREMENT")
        self.assertEqual(telomere["analyte_ids"], ["LEUKOCYTE_TELOMERE_LENGTH"])

    def test_pinned_consumer_tampering_fails_even_if_an_attacker_rehashes(self):
        self.source["consumer_config"]["theories"][0]["description"] = "Fabricated accepted theory"
        self.source["consumer_config_sha256"] = canonical_sha256(self.source["consumer_config"])
        with self.assertRaisesRegex(ValueError, "Pinned consumer_config"):
            export(self.source)

    def test_a_confirmed_prediction_cannot_cross_the_theory_bridge(self):
        self.source["predictions"][0]["confirmed"] = True
        with self.assertRaisesRegex(ValueError, "cannot be confirmed"):
            export(self.source)

    def test_empirical_relation_and_observation_references_are_rejected(self):
        for change in [{"relation": "consistent_with"}, {"observation_ids": ["OBS1"]}, {"empirical_edge": True}]:
            payload = copy.deepcopy(self.source)
            payload["links"][0].update(change)
            with self.assertRaises(ValueError):
                export(payload)

    def test_unknown_analyte_or_graph_node_fails_instead_of_silent_loss(self):
        self.source["predictions"][0]["analyte_ids"].append("INVENTED")
        with self.assertRaisesRegex(ValueError, "unknown theory or analyte"):
            export(self.source)
        self.source = json.loads(CATALOG.read_text())
        self.source["links"][0]["node_id"] = "L2_UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown theory or node"):
            export(self.source)

    def test_unchecked_bibliography_cannot_authorize_queries(self):
        self.source["articles"][0]["bibliographic_review"]["identity_matches"] = False
        with self.assertRaisesRegex(ValueError, "identity has not been checked"):
            export(self.source)

    def test_changed_doi_and_placeholder_sources_fail(self):
        self.source["articles"][0]["pmid"] = "999999"
        with self.assertRaisesRegex(ValueError, "URL/PMID mismatch"):
            export(self.source)
        self.source = json.loads(CATALOG.read_text())
        self.source["articles"][0]["doi"] = "10.1000/pretend"
        with self.assertRaises(ValueError):
            export(self.source)

    def test_direct_bytes_and_bibliographic_snapshot_hashes_are_checkable(self):
        pack = export_from_path(CATALOG, source_repository="owner/repo", source_commit=COMMIT)
        self.assertEqual(pack["provenance"]["registry_sha256"], hashlib.sha256(CATALOG.read_bytes()).hexdigest())
        manifest = json.loads((ROOT / "data/curated/bibliography/retrieval_manifest.json").read_text())
        raw = (ROOT / manifest["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["sha256"])
        records = {r["pmid"]: r for r in json.loads(raw)["resultList"]["result"]}
        self.assertEqual(set(records), set(manifest["expected_pmids"]))
        for source in self.source["articles"]:
            if source.get("pmid"):
                self.assertEqual(source["title"], records[source["pmid"]]["title"])
                self.assertEqual(source.get("doi"), records[source["pmid"]].get("doi"))


if __name__ == "__main__":
    unittest.main()
