import json
import logging
from pathlib import Path
from .metrics import MetricsCalculator, EvalResult, AggregatedMetrics
from .report import ReportGenerator

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, kgqa_service, config: dict):
        self.service = kgqa_service
        self.config = config

    def _load_eval_set(self, path: str) -> list[dict]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        sample = json.loads(line)
                        if all(k in sample for k in ["question", "gold_answers", "intent"]):
                            samples.append(sample)
                        else:
                            logger.warning(f"Line {line_num}: missing required fields, skipped")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: invalid JSON, skipped: {e}")
        return samples

    def run(self, eval_set_path: str) -> str:
        samples = self._load_eval_set(eval_set_path)
        logger.info(f"Loaded {len(samples)} evaluation samples")
        results = []
        for sample in samples:
            question = sample.get("question", "")
            gold_answers = sample.get("gold_answers", [])
            intent = sample.get("intent", "unknown")
            try:
                response = self.service.ask(question, top_k=10, mode="facts_only")  # 评测不调用LLM
                predicted = [a.text for a in response.answers]
                executable = bool(response.cypher)
                latency = response.latency_ms
            except Exception as e:
                logger.warning(f"Error processing question: {question}, error: {e}")
                predicted = []
                executable = False
                latency = 0
            precision, recall, f1 = MetricsCalculator.compute_set_metrics(gold_answers, predicted)
            exact_match = MetricsCalculator.exact_match(gold_answers, predicted)
            result = EvalResult(
                question=question,
                intent=intent,
                gold_answers=gold_answers,
                predicted_answers=predicted,
                executable=executable,
                latency_ms=latency,
                exact_match=exact_match,
                precision=precision,
                recall=recall,
                f1=f1,
            )
            results.append(result)
        metrics = MetricsCalculator.aggregate(results)
        slowest = sorted(results, key=lambda x: -x.latency_ms)[:10]
        failed = [r for r in results if not r.executable or not r.predicted_answers]
        report_path = ReportGenerator.generate(metrics, slowest, failed, self.config)
        logger.info(f"Evaluation complete. Report saved to {report_path}")
        return report_path
