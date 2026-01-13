#!/usr/bin/env python3
"""CM3KG 数据转换器：将 Disease.csv 和 medical.csv 转换为 triples JSONL 格式"""
import argparse
import ast
import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TYPE_MAPPING = {
    "疾病": "Disease",
    "症状": "Symptom",
    "药物": "Drug",
    "药品": "Drug",
    "检查": "Exam",
    "指标": "Indicator",
    "科室": "Department",
    "食物": "Food",
}

DISEASE_CSV_RELATION_MAP = {
    "症状": ("HAS_SYMPTOM", "Symptom"),
    "并发症": ("HAS_COMPLICATION", "Disease"),
    "推荐药品": ("RECOMMEND_DRUG", "Drug"),
    "常用药品": ("COMMON_DRUG", "Drug"),
    "宜吃食物": ("DO_EAT", "Food"),
    "忌吃食物": ("NOT_EAT", "Food"),
    "推荐食谱": ("RECOMMEND_EAT", "Food"),
    "一级科室分类": ("DEPT_L1", "Department"),
    "二级科室分类": ("DEPT_L2", "Department"),
    "三级科室分类": ("DEPT_L3", "Department"),
    "就诊科室": ("TREATED_IN", "Department"),
}

# 反向关系映射：需要交换 head 和 tail
DISEASE_CSV_REVERSE_RELATION_MAP = {
    "可能疾病": ("HAS_SYMPTOM", "Symptom", "Disease"),  # 症状->可能疾病->疾病 => 疾病-HAS_SYMPTOM->症状
}

DISEASE_CSV_PROPERTY_MAP = {
    "简介": "desc",
    "病因": "cause",
    "预防方式": "prevent",
    "治疗概述": "treatment_overview",
    "患病比例": "get_prob",
    "易感人群": "susceptible_pop",
    "传染方式": "transmission",
    "治疗方式": "cure_way",
    "治疗周期": "cure_time",
    "治愈率": "cure_rate",
    "治疗费用": "cost",
    "医保疾病": "is_insured",
}

MEDICAL_CSV_LIST_FIELDS = {
    "symptom": ("HAS_SYMPTOM", "Symptom"),
    "check": ("NEEDS_EXAM", "Exam"),
    "recommand_drug": ("RECOMMEND_DRUG", "Drug"),
    "common_drug": ("COMMON_DRUG", "Drug"),
    "cure_department": ("TREATED_IN", "Department"),
    "cure_way": ("HAS_TREATMENT", "Treatment"),
    "do_eat": ("DO_EAT", "Food"),
    "not_eat": ("NOT_EAT", "Food"),
    "recommand_eat": ("RECOMMEND_EAT", "Food"),
    "acompany": ("HAS_COMPLICATION", "Disease"),
}

MEDICAL_CSV_PROPERTY_FIELDS = [
    "desc", "prevent", "cause", "get_prob", "get_way",
    "cured_prob", "cure_lasttime", "cost_money", "yibao_status"
]


def detect_encoding(file_path: Path) -> str:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        try:
            with open(file_path, "r", encoding=enc) as f:
                for _ in range(100):
                    f.readline()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return "utf-8"


def parse_entity_with_type(raw: str) -> tuple[str, str]:
    """解析 'Name[Type]' 格式，返回 (name, mapped_type)"""
    match = re.match(r"^(.+)\[(.+)\]$", raw.strip())
    if match:
        name = match.group(1).strip()
        type_cn = match.group(2).strip()
        return name, TYPE_MAPPING.get(type_cn, "Entity")
    return raw.strip(), "Entity"


def safe_literal_eval(val: str) -> Optional[list]:
    if not val or not val.strip():
        return None
    try:
        result = ast.literal_eval(val)
        return result if isinstance(result, list) else None
    except (ValueError, SyntaxError):
        return None


class CM3KGConverter:
    def __init__(self, cm3kg_dir: Path, output_path: Path,
                 skip_properties: bool = False, limit: int = 0):
        self.cm3kg_dir = cm3kg_dir
        self.output_path = output_path
        self.skip_properties = skip_properties
        self.limit = limit
        self.seen_triples: set[tuple] = set()
        self.seen_properties: set[tuple] = set()
        self.triple_count = 0
        self.property_count = 0

    def _write_triple(self, head: str, head_type: str, relation: str,
                      tail: str, tail_type: str, source: str, confidence: float = 1.0):
        if not head or not tail or not relation:
            return False
        if self.limit > 0 and self.triple_count >= self.limit:
            return False
        key = (head, relation, tail)
        if key in self.seen_triples:
            return False
        self.seen_triples.add(key)
        triple = {
            "head": head,
            "head_type": head_type,
            "relation": relation,
            "tail": tail,
            "tail_type": tail_type,
            "source": source,
            "confidence": confidence,
        }
        self._out_file.write(json.dumps(triple, ensure_ascii=False) + "\n")
        self.triple_count += 1
        return True

    def _write_property(self, entity: str, entity_type: str, prop_name: str,
                        prop_value: str, source: str):
        if self.skip_properties:
            return
        if not entity or not prop_value or not prop_value.strip():
            return
        key = (entity, prop_name)
        if key in self.seen_properties:
            return
        self.seen_properties.add(key)
        prop_record = {
            "head": entity,
            "head_type": entity_type,
            "relation": f"PROP_{prop_name.upper()}",
            "tail": prop_value.strip(),
            "tail_type": "Text",
            "source": source,
            "confidence": 1.0,
            "is_property": True,
        }
        self._out_file.write(json.dumps(prop_record, ensure_ascii=False) + "\n")
        self.property_count += 1

    def process_disease_csv(self):
        disease_csv = self.cm3kg_dir / "Disease.csv"
        if not disease_csv.exists():
            logger.warning(f"Disease.csv not found: {disease_csv}")
            return

        encoding = detect_encoding(disease_csv)
        logger.info(f"Processing Disease.csv (encoding: {encoding})")

        with open(disease_csv, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if len(row) < 3:
                    continue
                raw_entity, predicate, value = row[0], row[1].strip(), row[2]

                head, head_type = parse_entity_with_type(raw_entity)
                if not head:
                    continue

                # 检查是否是属性
                if predicate in DISEASE_CSV_PROPERTY_MAP:
                    prop_name = DISEASE_CSV_PROPERTY_MAP[predicate]
                    self._write_property(head, head_type, prop_name, value, "CM3KG_DiseaseCSV")
                    continue

                # 检查是否是关系
                if predicate in DISEASE_CSV_RELATION_MAP:
                    relation, tail_type = DISEASE_CSV_RELATION_MAP[predicate]
                    # 处理可能包含多个值的情况（空格分隔）
                    if predicate in ("常用药品", "就诊科室", "并发症"):
                        tails = [t.strip() for t in re.split(r"\s+", value) if t.strip()]
                    else:
                        tails = [value.strip()] if value.strip() else []
                    for tail in tails:
                        self._write_triple(head, head_type, relation, tail, tail_type, "CM3KG_DiseaseCSV")
                    continue

                # 检查是否是反向关系（需要交换 head 和 tail）
                if predicate in DISEASE_CSV_REVERSE_RELATION_MAP:
                    relation, new_tail_type, new_head_type = DISEASE_CSV_REVERSE_RELATION_MAP[predicate]
                    # 原始: head[症状] -> 可能疾病 -> value[疾病]
                    # 转换: value[疾病] -> HAS_SYMPTOM -> head[症状]
                    new_head, _ = parse_entity_with_type(value)
                    if new_head:
                        self._write_triple(new_head, new_head_type, relation, head, new_tail_type, "CM3KG_DiseaseCSV")

                if row_num % 100000 == 0:
                    logger.info(f"Processed {row_num} rows from Disease.csv")

        logger.info(f"Disease.csv complete: {self.triple_count} triples, {self.property_count} properties")

    def process_medical_csv(self):
        medical_csv = self.cm3kg_dir / "medical.csv"
        if not medical_csv.exists():
            logger.warning(f"medical.csv not found: {medical_csv}")
            return

        encoding = detect_encoding(medical_csv)
        logger.info(f"Processing medical.csv (encoding: {encoding})")

        with open(medical_csv, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                head = row.get("name", "").strip()
                if not head:
                    continue
                head_type = "Disease"

                # 处理列表字段 -> 三元组
                for col, (relation, tail_type) in MEDICAL_CSV_LIST_FIELDS.items():
                    val = row.get(col, "")
                    items = safe_literal_eval(val)
                    if items:
                        for item in items:
                            if isinstance(item, str) and item.strip():
                                self._write_triple(head, head_type, relation,
                                                   item.strip(), tail_type, "CM3KG_MedicalCSV")

                # 处理属性字段
                for prop_col in MEDICAL_CSV_PROPERTY_FIELDS:
                    prop_val = row.get(prop_col, "")
                    if prop_val and prop_val.strip():
                        self._write_property(head, head_type, prop_col, prop_val, "CM3KG_MedicalCSV")

                if row_num % 5000 == 0:
                    logger.info(f"Processed {row_num} rows from medical.csv")

        logger.info(f"medical.csv complete: total {self.triple_count} triples, {self.property_count} properties")

    def run(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            self._out_file = f
            self.process_disease_csv()
            self.process_medical_csv()
        logger.info(f"Output saved to {self.output_path}")
        logger.info(f"Total: {self.triple_count} triples, {self.property_count} properties")


def main():
    parser = argparse.ArgumentParser(description="Convert CM3KG to triples JSONL")
    parser.add_argument("--cm3kg_dir", type=str, default="data/CM3KG",
                        help="Path to CM3KG directory")
    parser.add_argument("--out", type=str, default="data/triples_raw/cm3kg_triples.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--skip-properties", action="store_true",
                        help="Skip property records (desc, cause, prevent, etc.)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of triples (0 = no limit)")
    args = parser.parse_args()

    converter = CM3KGConverter(
        Path(args.cm3kg_dir), Path(args.out),
        skip_properties=args.skip_properties, limit=args.limit
    )
    converter.run()


if __name__ == "__main__":
    main()
