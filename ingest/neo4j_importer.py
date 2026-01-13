import logging
from datetime import datetime
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

ENTITY_TYPES = {
    "disease": "Disease", "疾病": "Disease",
    "symptom": "Symptom", "症状": "Symptom",
    "drug": "Drug", "药物": "Drug", "药品": "Drug",
    "exam": "Exam", "检查": "Exam",
    "indicator": "Indicator", "指标": "Indicator",
    "food": "Food", "食物": "Food",
    "department": "Department", "科室": "Department",
    "treatment": "Treatment", "治疗": "Treatment",
    "text": "Text",
}

RELATION_MAPPING = {
    "has_symptom": ("Disease", "HAS_SYMPTOM", "Symptom"),
    "症状": ("Disease", "HAS_SYMPTOM", "Symptom"),
    "recommended_drug": ("Disease", "RECOMMENDED_DRUG", "Drug"),
    "推荐药物": ("Disease", "RECOMMENDED_DRUG", "Drug"),
    "recommend_drug": ("Disease", "RECOMMEND_DRUG", "Drug"),
    "common_drug": ("Disease", "COMMON_DRUG", "Drug"),
    "treats": ("Drug", "TREATS", "Disease"),
    "治疗": ("Drug", "TREATS", "Disease"),
    "needs_exam": ("Disease", "NEEDS_EXAM", "Exam"),
    "检查": ("Disease", "NEEDS_EXAM", "Exam"),
    "has_indicator": ("Exam", "HAS_INDICATOR", "Indicator"),
    "指标": ("Exam", "HAS_INDICATOR", "Indicator"),
    "contraindicated_for": ("Drug", "CONTRAINDICATED_FOR", "Disease"),
    "禁忌": ("Drug", "CONTRAINDICATED_FOR", "Disease"),
    "related_to": (None, "RELATED_TO", None),
    "has_complication": ("Disease", "HAS_COMPLICATION", "Disease"),
    "do_eat": ("Disease", "DO_EAT", "Food"),
    "not_eat": ("Disease", "NOT_EAT", "Food"),
    "recommend_eat": ("Disease", "RECOMMEND_EAT", "Food"),
    "dept_l1": ("Disease", "DEPT_L1", "Department"),
    "dept_l2": ("Disease", "DEPT_L2", "Department"),
    "dept_l3": ("Disease", "DEPT_L3", "Department"),
    "treated_in": ("Disease", "TREATED_IN", "Department"),
    "has_treatment": ("Disease", "HAS_TREATMENT", "Treatment"),
}

ALLOWED_LABELS = {"Disease", "Symptom", "Drug", "Exam", "Indicator", "Entity",
                  "Food", "Department", "Treatment", "Text"}
ALLOWED_REL_TYPES = {"HAS_SYMPTOM", "RECOMMENDED_DRUG", "TREATS", "NEEDS_EXAM",
                     "HAS_INDICATOR", "CONTRAINDICATED_FOR", "RELATED_TO",
                     "RECOMMEND_DRUG", "COMMON_DRUG", "HAS_COMPLICATION",
                     "DO_EAT", "NOT_EAT", "RECOMMEND_EAT",
                     "DEPT_L1", "DEPT_L2", "DEPT_L3", "TREATED_IN", "HAS_TREATMENT"}

PROPERTY_RELATIONS = {
    "PROP_DESC": "desc",
    "PROP_CAUSE": "cause",
    "PROP_PREVENT": "prevent",
    "PROP_TREATMENT_OVERVIEW": "treatment_overview",
    "PROP_GET_PROB": "get_prob",
    "PROP_SUSCEPTIBLE_POP": "susceptible_pop",
    "PROP_TRANSMISSION": "transmission",
    "PROP_CURE_WAY": "cure_way",
    "PROP_CURE_TIME": "cure_time",
    "PROP_CURE_RATE": "cure_rate",
    "PROP_COST": "cost",
    "PROP_IS_INSURED": "is_insured",
    "PROP_GET_WAY": "get_way",
    "PROP_CURED_PROB": "cured_prob",
    "PROP_CURE_LASTTIME": "cure_lasttime",
    "PROP_COST_MONEY": "cost_money",
    "PROP_YIBAO_STATUS": "yibao_status",
}


class Neo4jImporter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Exam) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Indicator) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Food) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Department) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Treatment) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE",
        ]
        with self.driver.session() as session:
            for c in constraints:
                try:
                    session.run(c)
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
        # 为所有标签创建索引（如果约束不存在的话索引会帮助查询）
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Disease) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Drug) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Exam) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Indicator) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Food) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Department) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Treatment) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        ]
        with self.driver.session() as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception as e:
                    logger.warning(f"Index creation: {e}")

    def close(self):
        self.driver.close()

    def _infer_labels(self, relation: str) -> tuple[str, str, str]:
        rel_lower = relation.lower()
        if rel_lower in RELATION_MAPPING:
            head_label, rel_type, tail_label = RELATION_MAPPING[rel_lower]
            head_label = head_label or "Entity"
            tail_label = tail_label or "Entity"
            if head_label not in ALLOWED_LABELS:
                head_label = "Entity"
            if tail_label not in ALLOWED_LABELS:
                tail_label = "Entity"
            if rel_type not in ALLOWED_REL_TYPES:
                rel_type = "RELATED_TO"
            return head_label, rel_type, tail_label
        return "Entity", "RELATED_TO", "Entity"

    def import_fused_triples(self, triples: list[dict], alias_map: dict = None, batch_size: int = 30):
        alias_map = alias_map or {}
        canonical_aliases = alias_map.get("alias_map", {})
        now = datetime.now().isoformat()

        # 分离属性和关系
        property_records = []
        relation_batches = {}  # (head_label, rel_type, tail_label) -> list of records

        for t in triples:
            head = t.get("head", "")
            tail = t.get("tail", "")
            relation = t.get("relation", "RELATED_TO")
            if not head or not tail:
                continue

            if relation in PROPERTY_RELATIONS:
                prop_name = PROPERTY_RELATIONS[relation]
                head_type = t.get("head_type", "Entity")
                if head_type not in ALLOWED_LABELS:
                    head_type = "Entity"
                value = tail.strip()
                if value:
                    property_records.append({
                        "name": head, "head_type": head_type,
                        "prop_name": prop_name, "value": value
                    })
            else:
                head_type = t.get("head_type")
                tail_type = t.get("tail_type")
                if head_type and tail_type and head_type in ALLOWED_LABELS and tail_type in ALLOWED_LABELS:
                    head_label, tail_label = head_type, tail_type
                    rel_type = relation if relation in ALLOWED_REL_TYPES else "RELATED_TO"
                else:
                    head_label, rel_type, tail_label = self._infer_labels(relation)
                key = (head_label, rel_type, tail_label)
                if key not in relation_batches:
                    relation_batches[key] = []
                relation_batches[key].append({
                    "head": head, "tail": tail,
                    "head_aliases": canonical_aliases.get(head, [head]),
                    "tail_aliases": canonical_aliases.get(tail, [tail]),
                    "source": ",".join(t.get("sources", [t.get("source", "unknown")])),
                    "confidence": t.get("confidence", 1.0),
                    "frequency": t.get("frequency", 1),
                    "edge_score": t.get("edge_score", 0.0),
                    "conflict": t.get("conflict", False),
                })

        # 批量导入属性（按 head_type 和 prop_name 分组）
        prop_groups = {}
        for p in property_records:
            key = (p["head_type"], p["prop_name"])
            if key not in prop_groups:
                prop_groups[key] = []
            prop_groups[key].append({"name": p["name"], "value": p["value"]})

        property_count = 0
        logger.info(f"Starting property import: {len(prop_groups)} groups, {len(property_records)} total")
        with self.driver.session() as session:
            for (head_type, prop_name), records in prop_groups.items():
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    query = f"""
                    UNWIND $batch AS row
                    MERGE (n:{head_type} {{name: row.name}})
                    ON CREATE SET n.created_at = $now, n.updated_at = $now
                    SET n.{prop_name} = CASE
                        WHEN n.{prop_name} IS NULL THEN row.value
                        WHEN size(n.{prop_name}) < size(row.value) THEN row.value
                        ELSE n.{prop_name}
                    END, n.updated_at = $now
                    """
                    session.run(query, batch=batch, now=now)
                    property_count += len(batch)
                    if property_count % 10000 == 0:
                        logger.info(f"Imported {property_count} properties...")

        # 批量导入关系
        relation_count = 0
        total_relations = sum(len(r) for r in relation_batches.values())
        logger.info(f"Starting relation import: {len(relation_batches)} groups, {total_relations} total")
        with self.driver.session() as session:
            for (head_label, rel_type, tail_label), records in relation_batches.items():
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    query = f"""
                    UNWIND $batch AS row
                    MERGE (h:{head_label} {{name: row.head}})
                    ON CREATE SET h.aliases = row.head_aliases, h.created_at = $now, h.updated_at = $now
                    ON MATCH SET h.updated_at = $now
                    MERGE (t:{tail_label} {{name: row.tail}})
                    ON CREATE SET t.aliases = row.tail_aliases, t.created_at = $now, t.updated_at = $now
                    ON MATCH SET t.updated_at = $now
                    MERGE (h)-[r:{rel_type}]->(t)
                    SET r.source = row.source, r.confidence = row.confidence,
                        r.frequency = row.frequency, r.edge_score = row.edge_score,
                        r.conflict = row.conflict, r.predicted = false, r.created_at = $now
                    """
                    session.run(query, batch=batch, now=now)
                    relation_count += len(batch)
                    if relation_count % 10000 == 0:
                        logger.info(f"Imported {relation_count} relationships...")

        logger.info(f"Import complete: {relation_count} relationships, {property_count} properties")

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
        return {"nodes": node_count, "relationships": rel_count}
