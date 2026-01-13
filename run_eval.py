#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from app.neo4j_client import Neo4jClient
from app.kgqa_service import KGQAService
from eval.generator import EvalGenerator
from eval.runner import EvalRunner
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    neo4j_cfg = config.get("neo4j", {})
    client = Neo4jClient(neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"])

    logger.info("Step 1: Generating evaluation set...")
    generator = EvalGenerator(client, config)
    eval_path = str(Path(__file__).parent / "data" / "eval" / "eval_set.jsonl")
    samples_per_type = config.get("eval", {}).get("samples_per_type", 50)
    count = generator.generate(eval_path, samples_per_type)
    logger.info(f"Generated {count} evaluation samples")

    logger.info("Step 2: Running evaluation...")
    service = KGQAService(client)
    runner = EvalRunner(service, config)
    report_path = runner.run(eval_path)
    logger.info(f"Evaluation complete. Report: {report_path}")

    client.close()
    logger.info("Auto eval complete!")


if __name__ == "__main__":
    main()
