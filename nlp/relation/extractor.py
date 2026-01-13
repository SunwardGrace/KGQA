import json
import logging
import re
from pathlib import Path
from typing import Optional
from .schemas import TripleRecord
from ..ner.schemas import EntitySpan

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "schema" / "medical_schema.json"

RELATION_MAP = {
    # HAS_SYMPTOM
    "症状": "HAS_SYMPTOM", "表现": "HAS_SYMPTOM", "临床表现": "HAS_SYMPTOM", "相关症状": "HAS_SYMPTOM",
    # RECOMMENDED_DRUG
    "治疗药物": "RECOMMENDED_DRUG", "推荐药物": "RECOMMENDED_DRUG", "用药": "RECOMMENDED_DRUG",
    # TREATS
    "治疗": "TREATS", "适应症": "TREATS", "主治": "TREATS", "功能主治": "TREATS",
    # NEEDS_EXAM
    "检查": "NEEDS_EXAM", "检查项目": "NEEDS_EXAM", "诊断检查": "NEEDS_EXAM",
    # HAS_INDICATOR
    "指标": "HAS_INDICATOR", "检验指标": "HAS_INDICATOR",
    # CONTRAINDICATED_FOR
    "禁忌": "CONTRAINDICATED_FOR", "禁用": "CONTRAINDICATED_FOR",
    # CAUSES
    "病因": "CAUSES", "致病因素": "CAUSES",
    # COMPLICATION
    "并发症": "COMPLICATION", "合并症": "COMPLICATION",
    # RELATED_DISEASE
    "相关疾病": "RELATED_DISEASE", "关联疾病": "RELATED_DISEASE",
    # AFFECTS_BODY
    "累及部位": "AFFECTS_BODY", "发病部位": "AFFECTS_BODY",
    # DOSAGE
    "用法用量": "DOSAGE", "剂量": "DOSAGE",
    # ADVERSE_REACTION
    "不良反应": "ADVERSE_REACTION", "副作用": "ADVERSE_REACTION",
    # COMPOSITION
    "成份": "COMPOSITION", "组成": "COMPOSITION",
    # DRUG_INTERACTION
    "药物相互作用": "DRUG_INTERACTION", "相互作用": "DRUG_INTERACTION",
    # PRECAUTION
    "注意事项": "PRECAUTION",
    # BELONGS_TO
    "所属科室": "BELONGS_TO", "就诊科室": "BELONGS_TO",
    # PERFORMED_BY
    "执行科室": "PERFORMED_BY",
    # USES_EQUIPMENT
    "使用设备": "USES_EQUIPMENT",
    # CAUSED_BY_MICROBE
    "致病微生物": "CAUSED_BY_MICROBE", "病原体": "CAUSED_BY_MICROBE",
    # EFFICACY
    "功效": "EFFICACY", "药效": "EFFICACY",
    # PROPERTY
    "性味": "PROPERTY", "药性": "PROPERTY",
    # STORAGE
    "贮藏": "STORAGE", "保存方法": "STORAGE",
    # RELATED_TO
    "相关": "RELATED_TO", "关联": "RELATED_TO",
}

def load_schema() -> dict:
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load schema: {e}")
    return {}


def build_relation_map_from_schema(schema: dict) -> dict:
    rel_map = dict(RELATION_MAP)
    for rel_type in schema.get("relation_types", []):
        rel_id = rel_type.get("id", "")
        for alias in rel_type.get("aliases", []):
            rel_map[alias] = rel_id
    return rel_map


RELATION_PATTERNS = [
    # Disease -> Symptom
    (r"(.+?)的(症状|表现|临床表现)(?:有|包括|是)?(.+)", "HAS_SYMPTOM", "Disease", "Symptom"),
    (r"(.+?)(?:常见|可见|会出现|主要表现为)(.+?)(?:症状|表现)", "HAS_SYMPTOM", "Disease", "Symptom"),
    (r"患(.+?)(?:时)?(?:会|可能)(?:有|出现)(.+)", "HAS_SYMPTOM", "Disease", "Symptom"),
    # Disease -> Drug
    (r"(.+?)(?:可以)?(?:用|服用|吃)(.+?)(?:治疗|来治)?", "RECOMMENDED_DRUG", "Disease", "Drug"),
    (r"(.+?)的(?:治疗|常用)?药物(?:有|包括)?(.+)", "RECOMMENDED_DRUG", "Disease", "Drug"),
    (r"(?:治疗|缓解)(.+?)(?:可以|应该)?(?:用|服用)(.+)", "RECOMMENDED_DRUG", "Disease", "Drug"),
    # Drug -> Disease (TREATS)
    (r"(.+?)(?:用于|主治|适用于|可治疗)(.+)", "TREATS", "Drug", "Disease"),
    (r"(.+?)的(?:适应症|功能主治)(?:是|为|包括)?(.+)", "TREATS", "Drug", "Disease"),
    # Disease -> Exam
    (r"(.+?)(?:需要|应该)?(?:做|进行)(.+?)检查", "NEEDS_EXAM", "Disease", "Exam"),
    (r"(?:诊断|确诊)(.+?)(?:需要|应)?(?:做|进行)?(.+?)(?:检查)?", "NEEDS_EXAM", "Disease", "Exam"),
    (r"(.+?)的(?:检查|诊断)项目(?:有|包括)?(.+)", "NEEDS_EXAM", "Disease", "Exam"),
    # Exam -> Indicator
    (r"(.+?)(?:检查|检验)(?:的)?(?:指标|项目)(?:有|包括)?(.+)", "HAS_INDICATOR", "Exam", "Indicator"),
    # Drug contraindications
    (r"(.+?)(?:禁用|禁忌)(?:于|用于)?(.+)", "CONTRAINDICATED_FOR", "Drug", "Disease"),
    (r"(.+?)患者(?:禁用|不宜用|慎用)(.+)", "CONTRAINDICATED_FOR", "Disease", "Drug"),
    # Drug adverse reactions
    (r"(.+?)(?:的)?(?:不良反应|副作用)(?:有|包括)?(.+)", "ADVERSE_REACTION", "Drug", "Symptom"),
    (r"(?:服用|使用)(.+?)(?:可能|会)?(?:导致|引起|出现)(.+)", "ADVERSE_REACTION", "Drug", "Symptom"),
    # Disease -> Body
    (r"(.+?)(?:累及|影响|发生在|病变部位为?)(.+?)(?:部位)?", "AFFECTS_BODY", "Disease", "Body"),
    (r"(.+?)(?:是)?(.+?)(?:的|部位的)?疾病", "AFFECTS_BODY", "Disease", "Body"),
    # Disease complications
    (r"(.+?)(?:的)?(?:并发症|合并症)(?:有|包括)?(.+)", "COMPLICATION", "Disease", "Disease"),
    (r"(.+?)(?:可能|容易)?(?:并发|合并|引起)(.+)", "COMPLICATION", "Disease", "Disease"),
    # Disease causes
    (r"(.+?)(?:的)?(?:病因|致病因素|原因)(?:是|有|包括)?(.+)", "CAUSES", "Entity", "Disease"),
    (r"(.+?)(?:可以|会|能)?(?:导致|引起|引发)(.+)", "CAUSES", "Entity", "Disease"),
    # Disease -> Department
    (r"(.+?)(?:应该|需要)?(?:去|看|挂)(.+?)(?:科)?", "BELONGS_TO", "Disease", "Department"),
    (r"(.+?)(?:属于|归属)?(.+?)(?:科室)?(?:治疗|诊治)?", "BELONGS_TO", "Disease", "Department"),
    # Disease -> Microbe
    (r"(.+?)(?:由|是由)?(.+?)(?:引起|导致|感染)", "CAUSED_BY_MICROBE", "Disease", "Microbe"),
    (r"(.+?)(?:感染|侵入)(?:可)?(?:导致|引起)(.+)", "CAUSED_BY_MICROBE", "Microbe", "Disease"),
    # Drug interactions
    (r"(.+?)(?:与|和)(.+?)(?:有|存在)?(?:相互作用|配伍禁忌)", "DRUG_INTERACTION", "Drug", "Drug"),
    (r"(.+?)(?:不宜|不能)(?:与|和)(.+?)(?:同服|合用|同时使用)", "DRUG_INTERACTION", "Drug", "Drug"),
    # Drug dosage
    (r"(.+?)(?:的)?(?:用法用量|剂量|服用方法)(?:是|为)?(.+)", "DOSAGE", "Drug", "Entity"),
    # Drug composition
    (r"(.+?)(?:的)?(?:成份|成分|组成)(?:有|包括|是)?(.+)", "COMPOSITION", "Drug", "Entity"),
    # Drug efficacy
    (r"(.+?)(?:的)?(?:功效|药效|作用)(?:是|有|包括)?(.+)", "EFFICACY", "Drug", "Entity"),
    # Drug properties
    (r"(.+?)(?:的)?(?:性味|药性)(?:是|为)?(.+)", "PROPERTY", "Drug", "Entity"),
]


class RelationExtractor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        min_score: float = 0.5,
        relation_map: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.min_score = min_score
        self.model = None

        schema = load_schema()
        self.relation_map = relation_map or build_relation_map_from_schema(schema)

        if model_name:
            self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline

            self.model = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
            )
            logger.info(f"RE model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load RE model: {e}. Using rule-based fallback.")
            self.model = None

    def extract(
        self, text: str, entities: Optional[list[EntitySpan]] = None
    ) -> list[TripleRecord]:
        if not text:
            return []

        if self.model is not None and entities:
            return self._extract_with_model(text, entities)
        return self._extract_with_rules(text)

    def _extract_with_model(
        self, text: str, entities: list[EntitySpan]
    ) -> list[TripleRecord]:
        triples = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                if e1.label == e2.label:
                    continue
                context = text
                result = self.model(f"{e1.text}[SEP]{e2.text}[SEP]{context}")
                if result and result[0]["score"] >= self.min_score:
                    rel = self._map_relation(result[0]["label"])
                    if rel:
                        triples.append(
                            TripleRecord(
                                head=e1.text,
                                relation=rel,
                                tail=e2.text,
                                head_type=e1.label,
                                tail_type=e2.label,
                                confidence=result[0]["score"],
                                source="model",
                                evidence=text,
                                span={"head": [e1.start, e1.end], "tail": [e2.start, e2.end]},
                            )
                        )
        return triples

    def _extract_with_rules(self, text: str) -> list[TripleRecord]:
        triples = []
        for pattern, relation, head_type, tail_type in RELATION_PATTERNS:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                if len(groups) >= 2:
                    head = groups[0].strip()
                    tail = groups[-1].strip() if len(groups) == 2 else groups[2].strip()
                    tail_items = re.split(r"[,，、;；]", tail)
                    for item in tail_items:
                        item = item.strip()
                        if item and len(item) <= 20:
                            triples.append(
                                TripleRecord(
                                    head=head,
                                    relation=relation,
                                    tail=item,
                                    head_type=head_type,
                                    tail_type=tail_type,
                                    confidence=0.7,
                                    source="rule",
                                    evidence=text,
                                )
                            )
        return self._deduplicate(triples)

    def _map_relation(self, raw: str) -> Optional[str]:
        return self.relation_map.get(raw, raw if raw in RELATION_MAP.values() else None)

    def _deduplicate(self, triples: list[TripleRecord]) -> list[TripleRecord]:
        seen = set()
        result = []
        for t in triples:
            key = (t.head, t.relation, t.tail)
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result
