import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from .normalize import normalize_text
from .linking import EntityLinker
from .conflict import ConflictDetector
from .scoring import EdgeScorer, ScoredEdge

logger = logging.getLogger(__name__)

_nlp_ner = None
_nlp_re = None


def _get_ner_extractor(config: dict):
    global _nlp_ner
    if _nlp_ner is None:
        from nlp.ner import NERExtractor
        nlp_cfg = config.get("nlp", {})
        ner_cfg = nlp_cfg.get("ner", {})
        _nlp_ner = NERExtractor(
            model_name=ner_cfg.get("model", "bert-base-chinese"),
            device=ner_cfg.get("device", "cpu"),
            min_score=ner_cfg.get("min_score", 0.5),
        )
    return _nlp_ner


def _get_relation_extractor(config: dict):
    global _nlp_re
    if _nlp_re is None:
        from nlp.relation import RelationExtractor
        nlp_cfg = config.get("nlp", {})
        re_cfg = nlp_cfg.get("relation", {})
        _nlp_re = RelationExtractor(
            model_name=re_cfg.get("model"),
            device=re_cfg.get("device", "cpu"),
            min_score=re_cfg.get("min_score", 0.5),
        )
    return _nlp_re


class FusionPipeline:
    def __init__(self, config: dict):
        self.config = config
        fusion_cfg = config.get("fusion", {})
        self.fuzzy_threshold = fusion_cfg.get("fuzzy_threshold", 90)
        self.enable_fuzzy = fusion_cfg.get("enable_fuzzy", False)
        self.allow_multi = fusion_cfg.get("allow_multi_relations", [])
        self.source_weights = fusion_cfg.get("source_weights", {"default": 1.0})
        scoring_cfg = fusion_cfg.get("scoring", {})
        self.freq_weight = scoring_cfg.get("frequency_weight", 0.3)
        self.src_weight = scoring_cfg.get("source_weight", 0.3)
        self.conf_weight = scoring_cfg.get("confidence_weight", 0.4)
        self.linker = EntityLinker(self.fuzzy_threshold)
        self.conflict_detector = ConflictDetector(self.allow_multi)
        self.scorer = EdgeScorer(self.source_weights, self.freq_weight, self.src_weight, self.conf_weight)
        self.raw_triples: list[dict] = []
        self.fused_edges: list[ScoredEdge] = []

    def load_triples(self, file_path: str):
        path = Path(file_path)
        self.raw_triples = []
        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.raw_triples.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self.raw_triples = json.load(f)
        elif path.suffix == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.raw_triples = list(reader)
        logger.info(f"Loaded {len(self.raw_triples)} triples from {file_path}")

    def run(self) -> list[ScoredEdge]:
        entities = set()
        for t in self.raw_triples:
            entities.add(t.get("head", ""))
            entities.add(t.get("tail", ""))
        for e in entities:
            self.linker.add_entity(e)
        self.linker.merge_by_rules(list(entities))
        if self.enable_fuzzy:
            self.linker.merge_by_similarity(list(entities))
        self.linker.build_canonical_map()
        logger.info(f"Entity linking complete. {len(self.linker.canonical_map)} mappings created.")
        for t in self.raw_triples:
            head = self.linker.get_canonical(t.get("head", ""))
            tail = self.linker.get_canonical(t.get("tail", ""))
            relation = t.get("relation", "RELATED_TO")
            source = t.get("source", "unknown")
            try:
                confidence = float(t.get("confidence", 1.0))
            except (ValueError, TypeError):
                confidence = 1.0
            if head and tail and relation:
                self.conflict_detector.add_triple(head, relation, tail, source, confidence)
                self.scorer.add_edge(head, relation, tail, source, confidence)
        conflicts = self.conflict_detector.detect()
        logger.info(f"Conflict detection complete. {len(conflicts)} conflicts found.")
        conflict_map = self.conflict_detector.get_conflict_map()
        self.fused_edges = self.scorer.compute_scores(conflict_map)
        logger.info(f"Scoring complete. {len(self.fused_edges)} edges processed.")
        return self.fused_edges

    def export_fused_triples(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            for edge in self.fused_edges:
                record = {
                    "head": edge.head,
                    "relation": edge.relation,
                    "tail": edge.tail,
                    "frequency": edge.frequency,
                    "source_score": edge.source_score,
                    "confidence": edge.confidence,
                    "edge_score": edge.edge_score,
                    "conflict": edge.conflict,
                    "sources": edge.sources,
                    "created_at": now,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(self.fused_edges)} fused triples to {output_path}")

    def export_alias_mapping(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.linker.export_mapping(), f, ensure_ascii=False, indent=2)
        logger.info(f"Exported alias mapping to {output_path}")

    def export_conflict_report(self, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.conflict_detector.export_report(), f, ensure_ascii=False, indent=2)
        logger.info(f"Exported conflict report to {output_path}")

    def extract_from_text(self, text: str, source: str = "text_extraction") -> list[dict]:
        ner = _get_ner_extractor(self.config)
        re = _get_relation_extractor(self.config)
        entities = ner.extract(text)
        triples = re.extract(text, entities)
        result = []
        for t in triples:
            record = t.to_dict()
            record["source"] = source
            result.append(record)
            self.raw_triples.append(record)
        logger.info(f"Extracted {len(result)} triples from text")
        return result

    def extract_from_file(self, file_path: str, source: Optional[str] = None) -> list[dict]:
        path = Path(file_path)
        source = source or path.stem
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.extract_from_text(text, source)
