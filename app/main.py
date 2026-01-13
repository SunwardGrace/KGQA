import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import yaml
from fastapi import FastAPI
from app.api import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(title="KGQA - 中文医疗知识图谱问答系统", version="1.0.0")
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "KGQA API - 中文医疗知识图谱问答系统", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    api_cfg = config.get("api", {})
    uvicorn.run(app, host=api_cfg.get("host", "0.0.0.0"), port=api_cfg.get("port", 8000))
