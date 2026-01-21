import re
import time
import logging
from typing import Optional
from app.neo4j_client import Neo4jClient
from app.llm_client import LLMClient
from app.schemas import AskResponse, EntityMention, AnswerItem, SubGraph

logger = logging.getLogger(__name__)

INTENT_PATTERNS = [
    # symptom_to_disease 的具体模式优先
    (r"哪些疾病会出现(.+?)的症状", "symptom_to_disease"),
    (r"(哪些|什么)(疾病|病).*?症状.*?(.+)", "symptom_to_disease"),
    (r"(.+?)可能是什么(病|疾病)", "symptom_to_disease"),
    (r"(.+?)是.*?症状", "symptom_to_disease"),
    # disease_to_symptom 的通用模式放后面
    (r"(.+?)有(哪些|什么)(症状|表现)", "disease_to_symptom"),
    (r"(.+?)的症状", "disease_to_symptom"),
    (r"(.+?)吃(什么|哪些)药", "disease_to_drug"),
    (r"(.+?)的(治疗|用)药", "disease_to_drug"),
    (r"(什么|哪些)药.*?(治疗|治)(.+)", "drug_to_disease"),
    (r"(.+?)需要(做|哪些)(什么|哪些)?检查", "disease_to_exam"),
    (r"(.+?)的检查(项目)?", "disease_to_exam"),
    (r"(.+?)检查.*?(指标|项目)", "exam_to_indicator"),
    (r"(.+?)是什么", "entity_definition"),
    (r"介绍一下(.+)", "entity_definition"),
]

CYPHER_TEMPLATES = {
    "disease_to_symptom": """
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE toLower(d.name) = toLower($entity)
           OR toLower(d.name) CONTAINS toLower($entity)
           OR (d.aliases IS NOT NULL AND toLower($entity) IN [x IN d.aliases | toLower(x)])
        RETURN s.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               d.name as head, 'HAS_SYMPTOM' as relation
        ORDER BY CASE WHEN toLower(d.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "symptom_to_disease": """
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE toLower(s.name) = toLower($entity)
           OR toLower(s.name) CONTAINS toLower($entity)
           OR (s.aliases IS NOT NULL AND toLower($entity) IN [x IN s.aliases | toLower(x)])
        RETURN d.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               s.name as tail, 'HAS_SYMPTOM' as relation
        ORDER BY CASE WHEN toLower(s.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "disease_to_drug": """
        MATCH (d:Disease)-[r:RECOMMENDED_DRUG]->(dr:Drug)
        WHERE toLower(d.name) = toLower($entity)
           OR toLower(d.name) CONTAINS toLower($entity)
           OR (d.aliases IS NOT NULL AND toLower($entity) IN [x IN d.aliases | toLower(x)])
        RETURN dr.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               d.name as head, 'RECOMMENDED_DRUG' as relation
        ORDER BY CASE WHEN toLower(d.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "drug_to_disease": """
        MATCH (dr:Drug)-[r:TREATS]->(d:Disease)
        WHERE toLower(dr.name) = toLower($entity)
           OR toLower(dr.name) CONTAINS toLower($entity)
           OR (dr.aliases IS NOT NULL AND toLower($entity) IN [x IN dr.aliases | toLower(x)])
        RETURN d.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               dr.name as head, 'TREATS' as relation
        ORDER BY CASE WHEN toLower(dr.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "disease_to_exam": """
        MATCH (d:Disease)-[r:NEEDS_EXAM]->(e:Exam)
        WHERE toLower(d.name) = toLower($entity)
           OR toLower(d.name) CONTAINS toLower($entity)
           OR (d.aliases IS NOT NULL AND toLower($entity) IN [x IN d.aliases | toLower(x)])
        RETURN e.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               d.name as head, 'NEEDS_EXAM' as relation
        ORDER BY CASE WHEN toLower(d.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "exam_to_indicator": """
        MATCH (e:Exam)-[r:HAS_INDICATOR]->(i:Indicator)
        WHERE toLower(e.name) = toLower($entity)
           OR toLower(e.name) CONTAINS toLower($entity)
           OR (e.aliases IS NOT NULL AND toLower($entity) IN [x IN e.aliases | toLower(x)])
        RETURN i.name as answer, r.edge_score as score, r.source as source, r.conflict as conflict,
               e.name as head, 'HAS_INDICATOR' as relation
        ORDER BY CASE WHEN toLower(e.name) = toLower($entity) THEN 0 ELSE 1 END, r.edge_score DESC
        LIMIT $top_k
    """,
    "entity_definition": """
        MATCH (n)
        WHERE toLower(n.name) = toLower($entity)
           OR toLower(n.name) CONTAINS toLower($entity)
           OR (n.aliases IS NOT NULL AND toLower($entity) IN [x IN n.aliases | toLower(x)])
        RETURN n.name as answer, labels(n) as labels, n.aliases as aliases
        LIMIT 1
    """,
}

INTENT_ENTITY_TYPES = {
    "disease_to_symptom": "Disease",
    "symptom_to_disease": "Symptom",
    "disease_to_drug": "Disease",
    "drug_to_disease": "Drug",
    "disease_to_exam": "Disease",
    "exam_to_indicator": "Exam",
    "entity_definition": None,
}


class KGQAService:
    def __init__(self, neo4j_client: Neo4jClient, llm_client: Optional[LLMClient] = None):
        self.client = neo4j_client
        self.llm = llm_client

    def recognize_intent(self, query: str) -> tuple[str, str]:
        for pattern, intent in INTENT_PATTERNS:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                # 排除常见停用词
                stopwords = ["哪些", "什么", "疾病", "病", "治疗", "治", "做", "需要"]
                entity = next((g for g in groups if g and g not in stopwords), "")
                return intent, entity.strip()
        return "unknown", ""

    def extract_entity(self, query: str, intent: str, raw_entity: str) -> EntityMention | None:
        if not raw_entity:
            words = re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", query)
            raw_entity = max(words, key=len) if words else ""
        if not raw_entity:
            return None
        expected_type = INTENT_ENTITY_TYPES.get(intent)
        results = self.client.search_entity(raw_entity, expected_type)
        if results:
            best = results[0]
            return EntityMention(
                text=raw_entity,
                canonical=best["name"],
                type=best["labels"][0] if best["labels"] else "Entity",
            )
        return EntityMention(text=raw_entity, canonical=raw_entity, type=expected_type or "Entity")

    def generate_cypher(self, intent: str, entity: str, top_k: int) -> str:
        template = CYPHER_TEMPLATES.get(intent)
        if not template:
            return ""
        return template.strip()

    def execute_query(self, cypher: str, entity: str, top_k: int) -> list[dict]:
        if not cypher:
            return []
        return self.client.run_query(cypher, {"entity": entity, "top_k": top_k})

    def build_subgraph(self, results: list[dict], entity: str) -> SubGraph:
        nodes = []
        edges = []
        seen_nodes = set()
        for r in results:
            answer = r.get("answer", "")
            head = r.get("head", entity)
            tail = r.get("tail", answer)
            relation = r.get("relation", "RELATED_TO")
            if head and head not in seen_nodes:
                nodes.append({"id": head, "label": head})
                seen_nodes.add(head)
            if answer and answer not in seen_nodes:
                nodes.append({"id": answer, "label": answer})
                seen_nodes.add(answer)
            if head and answer:
                edges.append({"source": head, "target": answer, "relation": relation})
        return SubGraph(nodes=nodes, edges=edges)

    def ask(self, query: str, top_k: int = 10, mode: str = "smart") -> AskResponse:
        start = time.time()
        warnings = ["本系统仅用于知识检索与学习参考，不构成医疗建议。"]
        intent, raw_entity = self.recognize_intent(query)
        entity_mention = self.extract_entity(query, intent, raw_entity)
        if not entity_mention:
            return AskResponse(
                intent=intent,
                warnings=warnings + ["未能识别有效实体"],
                latency_ms=int((time.time() - start) * 1000),
            )
        cypher = self.generate_cypher(intent, entity_mention.canonical, top_k)
        results = self.execute_query(cypher, entity_mention.canonical, top_k)
        answers = []
        has_conflict = False
        for r in results:
            conflict = bool(r.get("conflict"))
            if conflict:
                has_conflict = True
            source_str = r.get("source", "")
            sources = source_str.split(",") if source_str else []
            score_val = r.get("score")
            answers.append(AnswerItem(
                text=r.get("answer", ""),
                score=score_val if score_val is not None else 1.0,
                source=sources,
                conflict=conflict,
            ))
        if has_conflict:
            warnings.append("检测到潜在冲突，部分答案来源不一致。")
        subgraph = self.build_subgraph(results, entity_mention.canonical)

        # LLM 生成自然语言回答
        llm_answer = ""
        if mode == "smart" and self.llm and self.llm.enabled:
            logger.info(f"Calling LLM for query: {query[:50]}...")
            answer_texts = [a.text for a in answers]
            entity_texts = [entity_mention.canonical]
            llm_answer = self.llm.generate(
                question=query,
                intent=intent,
                entities=entity_texts,
                answers=answer_texts,
            ) or ""
            logger.info(f"LLM response length: {len(llm_answer)}")
        elif mode == "smart":
            logger.warning(f"LLM not available: llm={self.llm}, enabled={self.llm.enabled if self.llm else 'N/A'}")

        latency = int((time.time() - start) * 1000)
        return AskResponse(
            intent=intent,
            parsed_entities=[entity_mention],
            cypher=cypher,
            answers=answers,
            subgraph=subgraph,
            llm_answer=llm_answer,
            warnings=warnings,
            latency_ms=latency,
        )
