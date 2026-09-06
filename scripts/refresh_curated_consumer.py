"""Refresh the validated derived consumer object after a curated-source edit."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from theories_pipeline.biomarker_bridge import export_theory_pack, _unique_object
from theories_pipeline.curated_catalog import build_consumer_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_unique_object)
    if payload.get("catalog_schema") != "curated_theory_discovery/1.0":
        parser.error("input must be a curated discovery catalog")
    payload.pop("consumer_config", None)
    legacy = deepcopy(payload)
    legacy.pop("catalog_schema")
    # This provisional pack is used only for fixture/quarantine validation;
    # it is never written or represented as a source-pinned export.
    pack = export_theory_pack(legacy, source_repository="validation-only",
                              source_commit="0" * 40, registry_sha256=hashlib.sha256(raw).hexdigest(),
                              input_file=args.input.as_posix())
    payload["consumer_config"] = build_consumer_config(payload, pack)
    args.input.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Validated and refreshed consumer_config; review and commit before exporting.")


if __name__ == "__main__":
    main()
