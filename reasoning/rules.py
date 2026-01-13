from dataclasses import dataclass


@dataclass
class Rule:
    rule_id: str
    description: str
    cypher: str
    target_relation: str


MEDICAL_RULES = [
    Rule(
        rule_id="r1_symptom_drug",
        description="If Disease has Symptom and Disease uses Drug, infer Drug may relieve Symptom",
        cypher="""
            MATCH (dis:Disease)-[:HAS_SYMPTOM]->(sym:Symptom)
            MATCH (dis)-[:RECOMMENDED_DRUG]->(drug:Drug)
            WHERE NOT (drug)-[:RELIEVES]->(sym)
            RETURN drug.name as head, 'RELIEVES' as relation, sym.name as tail,
                   dis.name as via, 0.6 as confidence
        """,
        target_relation="RELIEVES",
    ),
    Rule(
        rule_id="r2_disease_cooccurrence",
        description="If two diseases share multiple symptoms, they may be related",
        cypher="""
            MATCH (d1:Disease)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(d2:Disease)
            WHERE d1.name < d2.name
            WITH d1, d2, count(s) as shared
            WHERE shared >= 3
            RETURN d1.name as head, 'RELATED_DISEASE' as relation, d2.name as tail,
                   shared as evidence_count, 0.5 as confidence
        """,
        target_relation="RELATED_DISEASE",
    ),
    Rule(
        rule_id="r3_exam_chain",
        description="If Disease needs Exam and Exam has Indicator, link Disease to Indicator",
        cypher="""
            MATCH (dis:Disease)-[:NEEDS_EXAM]->(exam:Exam)-[:HAS_INDICATOR]->(ind:Indicator)
            WHERE NOT (dis)-[:MONITORS_INDICATOR]->(ind)
            RETURN dis.name as head, 'MONITORS_INDICATOR' as relation, ind.name as tail,
                   exam.name as via, 0.7 as confidence
        """,
        target_relation="MONITORS_INDICATOR",
    ),
    Rule(
        rule_id="r4_contraindication_propagation",
        description="If Drug A is contraindicated for Disease and Disease is related to Disease B, warn for B",
        cypher="""
            MATCH (drug:Drug)-[:CONTRAINDICATED_FOR]->(d1:Disease)
            MATCH (d1)-[:RELATED_DISEASE]-(d2:Disease)
            WHERE NOT (drug)-[:CONTRAINDICATED_FOR]->(d2)
            RETURN drug.name as head, 'POSSIBLE_CONTRAINDICATION' as relation, d2.name as tail,
                   d1.name as via, 0.4 as confidence
        """,
        target_relation="POSSIBLE_CONTRAINDICATION",
    ),
]
