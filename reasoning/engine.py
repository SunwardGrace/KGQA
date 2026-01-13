import logging
from typing import Optional
from datetime import datetime
from .rules import Rule, MEDICAL_RULES

logger = logging.getLogger(__name__)


class ReasoningEngine:
    def __init__(self, neo4j_client, rules: Optional[list[Rule]] = None, config: Optional[dict] = None):
        self.client = neo4j_client
        self.rules = rules or MEDICAL_RULES
        self.config = config or {}
        reasoning_cfg = self.config.get("reasoning", {})
        self.enabled = reasoning_cfg.get("enabled", True)
        self.max_inferences = reasoning_cfg.get("max_inferences", 1000)

    def run_rules(self, change_id: Optional[str] = None) -> list[dict]:
        if not self.enabled:
            logger.info("Reasoning engine disabled")
            return []

        all_inferences = []
        for rule in self.rules:
            try:
                results = self.client.run_query(rule.cypher, {})
                for r in results:
                    inference = {
                        "head": r.get("head"),
                        "relation": r.get("relation", rule.target_relation),
                        "tail": r.get("tail"),
                        "confidence": r.get("confidence", 0.5),
                        "rule_id": rule.rule_id,
                        "via": r.get("via"),
                        "predicted": True,
                        "change_id": change_id,
                    }
                    all_inferences.append(inference)
                logger.info(f"Rule {rule.rule_id}: {len(results)} inferences")
            except Exception as e:
                logger.warning(f"Rule {rule.rule_id} failed: {e}")

            if len(all_inferences) >= self.max_inferences:
                break

        return all_inferences

    def write_predictions(self, inferences: list[dict], change_id: Optional[str] = None) -> int:
        if not inferences:
            return 0

        now = datetime.now().isoformat()
        count = 0
        for inf in inferences:
            try:
                cypher = """
                    MATCH (h {name: $head})
                    MATCH (t {name: $tail})
                    MERGE (h)-[r:INFERRED {relation_type: $relation}]->(t)
                    SET r.confidence = $confidence,
                        r.rule_id = $rule_id,
                        r.via = $via,
                        r.predicted = true,
                        r.change_id = $change_id,
                        r.created_at = $now
                """
                self.client.run_query(cypher, {
                    "head": inf["head"],
                    "tail": inf["tail"],
                    "relation": inf["relation"],
                    "confidence": inf["confidence"],
                    "rule_id": inf["rule_id"],
                    "via": inf.get("via"),
                    "change_id": change_id,
                    "now": now,
                })
                count += 1
            except Exception as e:
                logger.warning(f"Failed to write inference: {e}")

        logger.info(f"Wrote {count} inferred edges to graph")
        return count

    def infer_for_entity(self, entity: str, top_k: int = 10) -> list[dict]:
        cypher = """
            MATCH (n {name: $entity})-[r:INFERRED]->(m)
            RETURN m.name as answer, r.relation_type as relation,
                   r.confidence as score, r.rule_id as rule_id, r.via as via
            ORDER BY r.confidence DESC
            LIMIT $top_k
            UNION
            MATCH (n)-[r:INFERRED]->(m {name: $entity})
            RETURN n.name as answer, r.relation_type as relation,
                   r.confidence as score, r.rule_id as rule_id, r.via as via
            ORDER BY r.confidence DESC
            LIMIT $top_k
        """
        return self.client.run_query(cypher, {"entity": entity, "top_k": top_k})
