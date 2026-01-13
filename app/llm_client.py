"""LLM 客户端：支持 OpenAI 兼容 API"""
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的医疗健康助手。根据知识图谱检索到的医学信息，为用户提供准确、专业的回答。

要求：
1. 基于提供的检索结果回答，不要编造信息
2. 用通俗易懂的语言解释医学术语
3. 在适当时提供健康建议
4. 始终提醒用户：具体诊疗请咨询专业医生

回答格式：
- 先直接回答问题
- 然后提供相关的健康建议（如有必要）
- 最后加上医疗免责声明"""


class LLMClient:
    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        timeout: float = 30.0,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enabled = bool(api_key)

        # 自动检测 API 端点格式
        if "/chat/completions" in self.api_base:
            self.endpoint = self.api_base
        else:
            self.endpoint = f"{self.api_base}/chat/completions"

        logger.info(f"LLM Client initialized: endpoint={self.endpoint}, model={self.model}, enabled={self.enabled}")

        if not self.enabled:
            logger.warning("LLM API key not configured, LLM generation disabled")

    def _build_prompt(self, question: str, intent: str, entities: list[str], answers: list[str]) -> str:
        """构造发送给 LLM 的用户 prompt"""
        context_parts = []

        if entities:
            context_parts.append(f"识别的实体: {', '.join(entities)}")

        if intent:
            intent_desc = {
                "disease_to_symptom": "查询疾病的症状",
                "symptom_to_disease": "根据症状查询可能的疾病",
                "disease_to_drug": "查询疾病的治疗药物",
                "disease_to_exam": "查询疾病需要的检查",
            }.get(intent, intent)
            context_parts.append(f"用户意图: {intent_desc}")

        if answers:
            context_parts.append(f"知识图谱检索结果: {', '.join(answers[:20])}")  # 限制数量
        else:
            context_parts.append("知识图谱检索结果: 未找到相关信息")

        context = "\n".join(context_parts)

        return f"""用户问题: {question}

{context}

请根据以上检索结果，为用户提供专业的回答和建议。"""

    def generate(
        self,
        question: str,
        intent: str = "",
        entities: Optional[list[str]] = None,
        answers: Optional[list[str]] = None,
    ) -> Optional[str]:
        """调用 LLM 生成回答"""
        if not self.enabled:
            return None

        entities = entities or []
        answers = answers or []

        user_prompt = self._build_prompt(question, intent, entities, answers)

        try:
            response = httpx.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            logger.error("LLM API timeout")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None
