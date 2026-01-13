from pydantic import BaseModel, Field


class EntityMention(BaseModel):
    text: str
    canonical: str
    type: str


class AnswerItem(BaseModel):
    text: str
    score: float = 1.0
    source: list[str] = Field(default_factory=list)
    conflict: bool = False


class SubGraph(BaseModel):
    nodes: list[dict] = Field(default_factory=list)
    edges: list[dict] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "smart"  # "facts_only" 或 "smart"（启用LLM）


class AskResponse(BaseModel):
    intent: str
    parsed_entities: list[EntityMention] = Field(default_factory=list)
    cypher: str = ""
    answers: list[AnswerItem] = Field(default_factory=list)
    subgraph: SubGraph = Field(default_factory=SubGraph)
    llm_answer: str = ""  # LLM 生成的自然语言回答
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int = 0


class EvalGenerateRequest(BaseModel):
    samples_per_type: int = 50


class EvalRunRequest(BaseModel):
    eval_set_path: str = "data/eval/eval_set.jsonl"


class EvalReportResponse(BaseModel):
    content: str
    generated_at: str = ""
