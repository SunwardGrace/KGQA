from dataclasses import dataclass


@dataclass
class EntitySpan:
    text: str
    start: int
    end: int
    label: str  # Disease|Symptom|Drug|Exam
    score: float = 1.0
