from dataclasses import dataclass, field


@dataclass
class EvalResult:
    question: str
    intent: str
    gold_answers: list[str]
    predicted_answers: list[str]
    executable: bool
    latency_ms: int
    exact_match: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class AggregatedMetrics:
    total_samples: int = 0
    executable_count: int = 0
    executable_rate: float = 0.0
    accuracy_at_1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    coverage: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    per_intent: dict = field(default_factory=dict)


class MetricsCalculator:
    @staticmethod
    def compute_set_metrics(gold: list[str], predicted: list[str]) -> tuple[float, float, float]:
        if not gold:
            return (1.0, 1.0, 1.0) if not predicted else (0.0, 0.0, 0.0)
        if not predicted:
            return 0.0, 0.0, 0.0
        gold_set = set(g.lower().strip() for g in gold)
        pred_set = set(p.lower().strip() for p in predicted)
        intersection = gold_set & pred_set
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def exact_match(gold: list[str], predicted: list[str]) -> bool:
        if not predicted:
            return False
        gold_set = set(g.lower().strip() for g in gold)
        return predicted[0].lower().strip() in gold_set

    @staticmethod
    def aggregate(results: list[EvalResult]) -> AggregatedMetrics:
        if not results:
            return AggregatedMetrics()
        total = len(results)
        executable = [r for r in results if r.executable]
        exec_count = len(executable)
        answered = [r for r in executable if r.predicted_answers]
        latencies = sorted([r.latency_ms for r in results])
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        per_intent = {}
        intent_groups = {}
        for r in results:
            if r.intent not in intent_groups:
                intent_groups[r.intent] = []
            intent_groups[r.intent].append(r)
        for intent, group in intent_groups.items():
            exec_in_group = [r for r in group if r.executable]
            answered_in_group = [r for r in exec_in_group if r.predicted_answers]
            per_intent[intent] = {
                "total": len(group),
                "executable_rate": len(exec_in_group) / len(group) if group else 0,
                "coverage": len(answered_in_group) / len(group) if group else 0,
                "avg_f1": sum(r.f1 for r in group) / len(group) if group else 0,
                "accuracy_at_1": sum(1 for r in group if r.exact_match) / len(group) if group else 0,
            }
        return AggregatedMetrics(
            total_samples=total,
            executable_count=exec_count,
            executable_rate=exec_count / total if total else 0,
            accuracy_at_1=sum(1 for r in results if r.exact_match) / total if total else 0,
            avg_precision=sum(r.precision for r in results) / total if total else 0,
            avg_recall=sum(r.recall for r in results) / total if total else 0,
            avg_f1=sum(r.f1 for r in results) / total if total else 0,
            coverage=len(answered) / total if total else 0,
            latency_p50=latencies[p50_idx] if latencies else 0,
            latency_p95=latencies[p95_idx] if p95_idx < len(latencies) else (latencies[-1] if latencies else 0),
            per_intent=per_intent,
        )
