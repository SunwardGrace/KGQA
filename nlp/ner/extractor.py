import json
import logging
from pathlib import Path
from typing import Optional
from .schemas import EntitySpan

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent.parent.parent / "data" / "schema" / "medical_schema.json"

LABEL_MAP = {
    "DISEASE": "Disease", "DIS": "Disease", "疾病": "Disease", "d": "Disease",
    "SYMPTOM": "Symptom", "SYM": "Symptom", "症状": "Symptom", "s": "Symptom",
    "DRUG": "Drug", "药物": "Drug", "y": "Drug",
    "EXAM": "Exam", "CHECK": "Exam", "检查": "Exam", "检验项目": "Exam", "i": "Exam",
    "BODY": "Body", "身体": "Body", "身体部位": "Body", "b": "Body",
    "PROCEDURE": "Procedure", "医疗程序": "Procedure", "p": "Procedure",
    "EQUIPMENT": "Equipment", "医疗设备": "Equipment", "e": "Equipment",
    "DEPARTMENT": "Department", "科室": "Department", "k": "Department",
    "MICROBE": "Microbe", "微生物": "Microbe", "微生物类": "Microbe", "m": "Microbe",
    "INDICATOR": "Indicator", "指标": "Indicator",
}

MEDICAL_ENTITIES = {
    "Disease": [
        "糖尿病", "高血压", "冠心病", "心肌梗死", "心力衰竭", "心律失常", "动脉硬化",
        "肺炎", "支气管炎", "哮喘", "慢性阻塞性肺病", "肺结核", "肺癌",
        "胃炎", "胃溃疡", "胃癌", "肝炎", "肝硬化", "肝癌", "脂肪肝",
        "肾炎", "肾衰竭", "尿毒症", "肾结石", "前列腺炎",
        "脑梗死", "脑出血", "帕金森病", "阿尔茨海默病", "癫痫", "偏头痛",
        "类风湿关节炎", "骨质疏松", "腰椎间盘突出", "颈椎病",
        "甲状腺功能亢进", "甲状腺功能减退", "甲状腺结节",
        "贫血", "白血病", "淋巴瘤", "艾滋病", "乙肝", "丙肝",
        "抑郁症", "焦虑症", "精神分裂症", "失眠症",
        "湿疹", "银屑病", "荨麻疹", "带状疱疹",
        "新冠肺炎", "流感", "感冒", "扁桃体炎", "中耳炎",
    ],
    "Symptom": [
        "发热", "咳嗽", "咳痰", "咯血", "胸闷", "胸痛", "心悸", "气短", "呼吸困难",
        "头痛", "头晕", "眩晕", "耳鸣", "视物模糊", "失眠", "嗜睡", "意识障碍",
        "恶心", "呕吐", "腹痛", "腹泻", "便秘", "腹胀", "食欲减退", "消化不良",
        "乏力", "疲劳", "体重减轻", "体重增加", "水肿", "浮肿",
        "多饮", "多尿", "多食", "口渴", "尿频", "尿急", "尿痛", "血尿",
        "皮疹", "瘙痒", "黄疸", "出血", "瘀斑", "关节痛", "肌肉痛", "腰痛",
        "麻木", "抽搐", "震颤", "瘫痪", "吞咽困难", "声音嘶哑",
    ],
    "Drug": [
        "阿司匹林", "布洛芬", "对乙酰氨基酚", "双氯芬酸", "吲哚美辛",
        "阿莫西林", "头孢克肟", "头孢曲松", "阿奇霉素", "左氧氟沙星", "青霉素",
        "二甲双胍", "格列美脲", "胰岛素", "阿卡波糖", "西格列汀",
        "氨氯地平", "硝苯地平", "卡托普利", "缬沙坦", "美托洛尔", "阿托伐他汀",
        "奥美拉唑", "雷贝拉唑", "法莫替丁", "多潘立酮", "莫沙必利",
        "氯雷他定", "西替利嗪", "地塞米松", "泼尼松", "甲泼尼龙",
        "阿托品", "山莨菪碱", "硝酸甘油", "华法林", "氯吡格雷",
        "甲硝唑", "替硝唑", "氟康唑", "伊曲康唑", "阿昔洛韦",
        "维生素C", "维生素D", "维生素B12", "叶酸", "铁剂", "钙片",
    ],
    "Exam": [
        "血常规", "尿常规", "便常规", "肝功能", "肾功能", "血糖", "血脂",
        "心电图", "动态心电图", "心脏彩超", "冠脉造影", "心肌酶谱",
        "胸部X光", "胸部CT", "头颅CT", "头颅MRI", "腹部B超", "腹部CT",
        "胃镜", "肠镜", "支气管镜", "膀胱镜",
        "甲状腺功能", "肿瘤标志物", "凝血功能", "电解质", "血气分析",
        "糖化血红蛋白", "餐后血糖", "葡萄糖耐量试验",
        "骨密度", "关节X光", "脊柱MRI",
    ],
    "Body": [
        "心脏", "肺", "肝脏", "肾脏", "胃", "肠", "大脑", "脊髓",
        "眼睛", "耳朵", "鼻子", "咽喉", "气管", "支气管",
        "血管", "动脉", "静脉", "淋巴结", "骨骼", "关节", "肌肉",
        "皮肤", "甲状腺", "胰腺", "脾脏", "胆囊", "膀胱", "前列腺",
    ],
    "Department": [
        "内科", "外科", "妇产科", "儿科", "急诊科", "ICU",
        "心内科", "呼吸内科", "消化内科", "神经内科", "内分泌科", "肾内科",
        "骨科", "泌尿外科", "神经外科", "心胸外科", "普外科",
        "皮肤科", "眼科", "耳鼻喉科", "口腔科", "康复科", "中医科",
    ],
}


def load_schema() -> dict:
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load schema: {e}")
    return {}


def build_label_map_from_schema(schema: dict) -> dict:
    label_map = dict(LABEL_MAP)
    for entity_type in schema.get("entity_types", []):
        entity_id = entity_type.get("id", "")
        for alias in entity_type.get("aliases", []):
            label_map[alias] = entity_id
            label_map[alias.upper()] = entity_id
    return label_map


class NERExtractor:
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        device: str = "cpu",
        label_map: Optional[dict] = None,
        min_score: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device
        self.min_score = min_score
        self.pipeline = None
        self.entity_type_map = None

        schema = load_schema()
        self.label_map = label_map or build_label_map_from_schema(schema)

        self._load_entity_type_map()
        self._load_model()

    def _load_entity_type_map(self):
        """Load entity type mapping from trained model directory."""
        model_path = Path(self.model_name)
        entity_map_path = model_path / "entity_type_map.json"
        if entity_map_path.exists():
            try:
                with open(entity_map_path, "r", encoding="utf-8") as f:
                    self.entity_type_map = json.load(f)
                logger.info(f"Loaded entity type map from {entity_map_path}")
            except Exception as e:
                logger.warning(f"Failed to load entity type map: {e}")

    def _load_model(self):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                device=self._resolve_device(),
                aggregation_strategy="simple",
            )
            logger.info(f"NER model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}. Using rule-based fallback.")
            self.pipeline = None

    def _resolve_device(self) -> int:
        if isinstance(self.device, int):
            return self.device
        if isinstance(self.device, str):
            device = self.device.strip().lower()
            if device == "cpu":
                return -1
            if device.isdigit():
                return int(device)
            if device.startswith("cuda"):
                parts = device.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    return int(parts[1])
                return 0
        return -1

    def _normalize_label(self, raw_label: str) -> str:
        label_clean = raw_label.upper()
        for prefix in ["B-", "I-", "E-", "S-", "M-"]:
            label_clean = label_clean.replace(prefix, "")
        label_lower = label_clean.lower()

        # Check CMeEE entity type map first (e.g., dis -> Disease)
        if self.entity_type_map and label_lower in self.entity_type_map:
            return self.entity_type_map[label_lower]

        # Fall back to schema label map
        return self.label_map.get(label_clean, self.label_map.get(raw_label, "Entity"))

    def extract(self, text: str) -> list[EntitySpan]:
        if not text:
            return []

        if self.pipeline is not None:
            return self._extract_with_model(text)
        return self._extract_with_rules(text)

    def _extract_with_model(self, text: str) -> list[EntitySpan]:
        try:
            results = self.pipeline(text, truncation=True)
        except Exception as e:
            logger.warning(f"NER inference failed: {e}. Using rule-based fallback.")
            return self._extract_with_rules(text)

        entities = []
        for r in results:
            score = r.get("score", 1.0)
            if score < self.min_score:
                continue
            label = self._normalize_label(r.get("entity_group", r.get("entity", "Entity")))
            entities.append(
                EntitySpan(
                    text=r.get("word", "").replace("##", ""),
                    start=r.get("start", 0),
                    end=r.get("end", 0),
                    label=label,
                    score=score,
                )
            )
        return self._merge_adjacent(entities)

    def _extract_with_rules(self, text: str) -> list[EntitySpan]:
        import re
        entities = []

        for entity_type, entity_list in MEDICAL_ENTITIES.items():
            pattern = "|".join(re.escape(e) for e in sorted(entity_list, key=len, reverse=True))
            if not pattern:
                continue
            for m in re.finditer(pattern, text):
                entities.append(
                    EntitySpan(
                        text=m.group(),
                        start=m.start(),
                        end=m.end(),
                        label=entity_type,
                        score=0.85,
                    )
                )

        entities = sorted(entities, key=lambda x: (x.start, -len(x.text)))
        return self._remove_overlapping(entities)

    def _remove_overlapping(self, entities: list[EntitySpan]) -> list[EntitySpan]:
        if not entities:
            return []
        result = []
        last_end = -1
        for e in entities:
            if e.start >= last_end:
                result.append(e)
                last_end = e.end
        return result

    def _merge_adjacent(self, entities: list[EntitySpan]) -> list[EntitySpan]:
        if not entities:
            return []
        entities = sorted(entities, key=lambda x: x.start)
        merged = [entities[0]]
        for e in entities[1:]:
            prev = merged[-1]
            if prev.label == e.label and e.start <= prev.end + 1:
                merged[-1] = EntitySpan(
                    text=prev.text + e.text,
                    start=prev.start,
                    end=e.end,
                    label=prev.label,
                    score=(prev.score + e.score) / 2,
                )
            else:
                merged.append(e)
        return merged
