from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TripleRecord:
    head: str
    relation: str
    tail: str
    head_type: str = "Entity"
    tail_type: str = "Entity"
    confidence: float = 1.0
    source: str = "extraction"
    evidence: Optional[str] = None
    span: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence,
            "span": self.span,
        }
