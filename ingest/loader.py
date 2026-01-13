import json
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TripleLoader:
    @staticmethod
    def load(file_path: str) -> list[dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        triples = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        triples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                triples = data if isinstance(data, list) else [data]
        elif path.suffix == ".csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                triples = list(reader)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        logger.info(f"Loaded {len(triples)} triples from {file_path}")
        return triples

    @staticmethod
    def load_fused(file_path: str) -> list[dict]:
        return TripleLoader.load(file_path)
