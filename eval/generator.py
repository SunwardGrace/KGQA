import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

QUESTION_TEMPLATES = {
    "disease_to_symptom": [
        "{entity}有哪些症状？",
        "{entity}的症状是什么？",
        "{entity}有什么表现？",
    ],
    "symptom_to_disease": [
        "哪些疾病会出现{entity}的症状？",
        "{entity}可能是什么病？",
    ],
    "disease_to_drug": [
        "{entity}吃什么药？",
        "{entity}的治疗药物有哪些？",
    ],
    "disease_to_exam": [
        "{entity}需要做哪些检查？",
        "{entity}的检查项目有哪些？",
    ],
}

INTENT_CYPHER = {
    "disease_to_symptom": """
        MATCH (d:Disease {name: $entity})-[:HAS_SYMPTOM]->(s:Symptom)
        RETURN s.name as answer
    """,
    "symptom_to_disease": """
        MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {name: $entity})
        RETURN d.name as answer
    """,
    "disease_to_drug": """
        MATCH (d:Disease {name: $entity})-[:RECOMMENDED_DRUG]->(dr:Drug)
        RETURN dr.name as answer
    """,
    "disease_to_exam": """
        MATCH (d:Disease {name: $entity})-[:NEEDS_EXAM]->(e:Exam)
        RETURN e.name as answer
    """,
}

INTENT_SAMPLE_QUERY = {
    "disease_to_symptom": "MATCH (d:Disease)-[:HAS_SYMPTOM]->() RETURN DISTINCT d.name as name LIMIT $limit",
    "symptom_to_disease": "MATCH ()-[:HAS_SYMPTOM]->(s:Symptom) RETURN DISTINCT s.name as name LIMIT $limit",
    "disease_to_drug": "MATCH (d:Disease)-[:RECOMMENDED_DRUG]->() RETURN DISTINCT d.name as name LIMIT $limit",
    "disease_to_exam": "MATCH (d:Disease)-[:NEEDS_EXAM]->() RETURN DISTINCT d.name as name LIMIT $limit",
}


class EvalGenerator:
    def __init__(self, neo4j_client, config: dict):
        self.client = neo4j_client
        self.config = config
        eval_cfg = config.get("eval", {})
        self.seed = eval_cfg.get("random_seed", 42)
        random.seed(self.seed)

    def _sample_entities(self, intent: str, limit: int) -> list[str]:
        query = INTENT_SAMPLE_QUERY.get(intent)
        if not query:
            return []
        results = self.client.run_query(query, {"limit": limit * 2})
        names = [r["name"] for r in results if r.get("name")]
        random.shuffle(names)
        return names[:limit]

    def _get_gold_answers(self, intent: str, entity: str) -> list[str]:
        cypher = INTENT_CYPHER.get(intent)
        if not cypher:
            return []
        results = self.client.run_query(cypher, {"entity": entity})
        return [r["answer"] for r in results if r.get("answer")]

    def generate(self, output_path: str, samples_per_type: int = 50) -> int:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = []
        for intent, templates in QUESTION_TEMPLATES.items():
            entities = self._sample_entities(intent, samples_per_type)
            for entity in entities:
                gold_answers = self._get_gold_answers(intent, entity)
                if not gold_answers:
                    continue
                template = random.choice(templates)
                question = template.format(entity=entity)
                sample = {
                    "question": question,
                    "intent": intent,
                    "entity_mentions": [entity],
                    "cypher": INTENT_CYPHER[intent].strip(),
                    "gold_answers": gold_answers,
                }
                samples.append(sample)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"Generated {len(samples)} evaluation samples to {output_path}")
        return len(samples)
