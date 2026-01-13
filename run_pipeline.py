#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
import json
import logging
from fusion import FusionPipeline
from ingest import Neo4jImporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = Path(__file__).parent / "data" / "triples_raw" / "cm3kg_triples.jsonl"
    fused_path = Path(__file__).parent / "data" / "triples_fused" / "fused_triples.jsonl"
    alias_path = Path(__file__).parent / "data" / "triples_fused" / "alias_mapping.json"
    conflict_path = Path(__file__).parent / "data" / "triples_fused" / "conflict_report.json"

    logger.info("Step 1: Running auto_fusion pipeline...")
    pipeline = FusionPipeline(config)
    pipeline.load_triples(str(raw_path))
    pipeline.run()
    pipeline.export_fused_triples(str(fused_path))
    pipeline.export_alias_mapping(str(alias_path))
    pipeline.export_conflict_report(str(conflict_path))

    logger.info("Step 2: Importing to Neo4j...")
    neo4j_cfg = config.get("neo4j", {})
    importer = Neo4jImporter(neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"])

    with open(alias_path, "r", encoding="utf-8") as f:
        alias_map = json.load(f)

    triples = []
    with open(fused_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triples.append(json.loads(line))

    importer.import_fused_triples(triples, alias_map)
    stats = importer.get_stats()
    logger.info(f"Import complete. Nodes: {stats['nodes']}, Relationships: {stats['relationships']}")
    importer.close()

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
