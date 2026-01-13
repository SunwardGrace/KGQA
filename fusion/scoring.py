from collections import defaultdict
from dataclasses import dataclass


@dataclass
class ScoredEdge:
    head: str
    relation: str
    tail: str
    frequency: int
    source_score: float
    confidence: float
    edge_score: float
    conflict: bool
    sources: list[str]


class EdgeScorer:
    def __init__(self, source_weights: dict[str, float] = None,
                 freq_weight: float = 0.3, src_weight: float = 0.3, conf_weight: float = 0.4):
        self.source_weights = source_weights or {"default": 1.0}
        self.freq_weight = freq_weight
        self.src_weight = src_weight
        self.conf_weight = conf_weight
        self.edge_data: dict[tuple, list[dict]] = defaultdict(list)

    def add_edge(self, head: str, relation: str, tail: str,
                 source: str = "default", confidence: float = 1.0):
        key = (head, relation, tail)
        self.edge_data[key].append({"source": source, "confidence": confidence})

    def compute_scores(self, conflict_map: dict[tuple, bool] = None) -> list[ScoredEdge]:
        conflict_map = conflict_map or {}
        results = []
        max_freq = max((len(v) for v in self.edge_data.values()), default=1)
        for (head, relation, tail), entries in self.edge_data.items():
            freq = len(entries)
            freq_norm = freq / max_freq if max_freq > 0 else 0
            sources = list(set(e["source"] for e in entries))
            src_score = sum(self.source_weights.get(s, self.source_weights.get("default", 1.0))
                            for s in sources) / len(sources) if sources else 0
            conf_avg = sum(e["confidence"] for e in entries) / len(entries) if entries else 0
            edge_score = (
                self.freq_weight * freq_norm +
                self.src_weight * min(src_score, 1.0) +
                self.conf_weight * conf_avg
            )
            is_conflict = conflict_map.get((head, relation), False)
            results.append(ScoredEdge(
                head=head, relation=relation, tail=tail,
                frequency=freq, source_score=src_score, confidence=conf_avg,
                edge_score=round(edge_score, 4), conflict=is_conflict, sources=sources,
            ))
        return sorted(results, key=lambda x: -x.edge_score)
