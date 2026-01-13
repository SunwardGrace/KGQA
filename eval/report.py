from datetime import datetime
from pathlib import Path
from .metrics import AggregatedMetrics, EvalResult


class ReportGenerator:
    @staticmethod
    def generate(metrics: AggregatedMetrics, slowest: list[EvalResult],
                 failed: list[EvalResult], config: dict) -> str:
        output_dir = Path(config.get("eval", {}).get("output_dir", "reports"))
        output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        report_lines = [
            "# KGQA 评测报告",
            "",
            f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 总体指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总样本数 | {metrics.total_samples} |",
            f"| 可执行率 | {metrics.executable_rate:.2%} |",
            f"| Accuracy@1 | {metrics.accuracy_at_1:.2%} |",
            f"| 平均Precision | {metrics.avg_precision:.4f} |",
            f"| 平均Recall | {metrics.avg_recall:.4f} |",
            f"| 平均F1 | {metrics.avg_f1:.4f} |",
            f"| Coverage | {metrics.coverage:.2%} |",
            f"| Latency P50 | {metrics.latency_p50:.0f}ms |",
            f"| Latency P95 | {metrics.latency_p95:.0f}ms |",
            "",
            "## 分意图指标",
            "",
            "| 意图 | 样本数 | 可执行率 | Accuracy@1 | 平均F1 | Coverage |",
            "|------|--------|----------|------------|--------|----------|",
        ]
        for intent, data in metrics.per_intent.items():
            report_lines.append(
                f"| {intent} | {data['total']} | {data['executable_rate']:.2%} | "
                f"{data['accuracy_at_1']:.2%} | {data['avg_f1']:.4f} | {data['coverage']:.2%} |"
            )
        report_lines.extend([
            "",
            "## 最慢 Top-10 样本",
            "",
            "| 问题 | 意图 | 延迟(ms) |",
            "|------|------|----------|",
        ])
        for r in slowest[:10]:
            q = r.question[:30] + "..." if len(r.question) > 30 else r.question
            report_lines.append(f"| {q} | {r.intent} | {r.latency_ms} |")
        report_lines.extend([
            "",
            f"## 失败样本统计: {len(failed)} 个",
            "",
        ])
        if failed:
            report_lines.append("| 问题 | 意图 | 原因 |")
            report_lines.append("|------|------|------|")
            for r in failed[:20]:
                q = r.question[:30] + "..." if len(r.question) > 30 else r.question
                reason = "查询不可执行" if not r.executable else "无结果返回"
                report_lines.append(f"| {q} | {r.intent} | {reason} |")
        report_content = "\n".join(report_lines)
        latest_path = output_dir / "latest_report.md"
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        timestamped_path = output_dir / f"report_{now.strftime('%Y%m%d_%H%M%S')}.md"
        with open(timestamped_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return str(latest_path)
