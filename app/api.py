from fastapi import APIRouter, HTTPException
from app.schemas import AskRequest, AskResponse, EvalGenerateRequest, EvalRunRequest, EvalReportResponse
from app.kgqa_service import KGQAService
from app.neo4j_client import Neo4jClient
from app.llm_client import LLMClient
import yaml
from pathlib import Path

router = APIRouter()

_service: KGQAService | None = None


def get_service() -> KGQAService:
    global _service
    if _service is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        neo4j_cfg = config.get("neo4j", {})
        client = Neo4jClient(neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"])

        # 初始化 LLM 客户端
        llm_cfg = config.get("llm", {})
        llm_client = None
        if llm_cfg.get("enabled", False):
            llm_client = LLMClient(
                api_base=llm_cfg.get("api_base", "https://api.openai.com/v1"),
                api_key=llm_cfg.get("api_key", ""),
                model=llm_cfg.get("model", "gpt-3.5-turbo"),
                max_tokens=llm_cfg.get("max_tokens", 1024),
                temperature=llm_cfg.get("temperature", 0.7),
                timeout=llm_cfg.get("timeout", 30.0),
            )

        _service = KGQAService(client, llm_client)
    return _service


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    service = get_service()
    return service.ask(request.query, request.top_k, request.mode)


@router.post("/eval/generate")
def eval_generate(request: EvalGenerateRequest):
    from eval.generator import EvalGenerator
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    neo4j_cfg = config.get("neo4j", {})
    client = Neo4jClient(neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"])
    generator = EvalGenerator(client, config)
    output_path = "data/eval/eval_set.jsonl"
    count = generator.generate(output_path, request.samples_per_type)
    return {"status": "success", "samples_generated": count, "output_path": output_path}


@router.post("/eval/run")
def eval_run(request: EvalRunRequest):
    from eval.runner import EvalRunner
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    service = get_service()
    runner = EvalRunner(service, config)
    report_path = runner.run(request.eval_set_path)
    return {"status": "success", "report_path": report_path}


@router.get("/eval/report", response_model=EvalReportResponse)
def eval_report():
    report_path = Path(__file__).parent.parent / "reports" / "latest_report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    content = report_path.read_text(encoding="utf-8")
    return EvalReportResponse(content=content)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stats")
def stats():
    service = get_service()
    return service.client.get_graph_stats()


@router.get("/changesets")
def list_changesets():
    from ingest.versioning import GraphVersioning
    service = get_service()
    versioning = GraphVersioning(service.client)
    return versioning.list_changesets()


@router.post("/changesets/{change_id}/rollback")
def rollback_changeset(change_id: str):
    from ingest.versioning import GraphVersioning
    service = get_service()
    versioning = GraphVersioning(service.client)
    try:
        count = versioning.rollback(change_id)
        return {"success": True, "count": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/entity/{entity_name}/history")
def entity_history(entity_name: str):
    from ingest.versioning import GraphVersioning
    service = get_service()
    versioning = GraphVersioning(service.client)
    return versioning.get_entity_history(entity_name)
