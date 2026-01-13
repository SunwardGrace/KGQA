from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ConflictRecord:
    head: str
    relation: str
    tails: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    is_conflict: bool = False


class ConflictDetector:
    def __init__(self, allow_multi_relations: list[str] = None):
        self.allow_multi = set(allow_multi_relations or [])
        self.relation_groups: dict[tuple, list[dict]] = defaultdict(list)
        self.conflicts: list[ConflictRecord] = []

    def add_triple(self, head: str, relation: str, tail: str,
                   source: str = "unknown", confidence: float = 1.0):
        key = (head, relation)
        self.relation_groups[key].append({
            "tail": tail,
            "source": source,
            "confidence": confidence,
        })

    def detect(self) -> list[ConflictRecord]:
        self.conflicts = []
        for (head, relation), entries in self.relation_groups.items():
            unique_tails = set(e["tail"] for e in entries)
            if len(unique_tails) <= 1:
                continue
            if relation.upper() in self.allow_multi:
                continue
            record = ConflictRecord(
                head=head,
                relation=relation,
                tails=list(unique_tails),
                sources=list(set(e["source"] for e in entries)),
                is_conflict=True,
            )
            self.conflicts.append(record)
        return self.conflicts

    def get_conflict_map(self) -> dict[tuple, bool]:
        return {(c.head, c.relation): True for c in self.conflicts}

    def export_report(self) -> list[dict]:
        return [
            {
                "head": c.head,
                "relation": c.relation,
                "tails": c.tails,
                "sources": c.sources,
            }
            for c in self.conflicts
        ]
