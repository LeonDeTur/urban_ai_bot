from typing import Any

import pandas as pd
import requests
from loguru import logger


from src.common.config.config import Config


class LlmService:
    def __init__(self, config: Config):

        self.url = f"http://{config.get('LLM_HOST')}:{config.get('LLM_PORT')}"
        self.model_name = config.get('LLM_MODEL')
        self.client_cert = config.get("CLIENT_CERT")

    async def generate_response(self, headers: dict, data: dict) -> str | None:

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                headers=headers,
                json=data,
            )
            return response.json()["response"]
        except Exception as e:
            logger.exception(e)
            return None

    async def generate_request_data(self, message: str, context: str, stream: bool=True) -> tuple[dict, dict]:
        data = {
            "model": self.model_name,
            "prompt": f"НАЧАЛО ВОПРОСА | {message} | КОНЕЦ ВОПРОСА",
            "stream": stream,
            "system": "Ты отвечаешь на вопросы по документам, связанным с градостроительством и урбанистикой \"СП 42.13330.2016 Градостроительство. Планировка и застройка городских и сельских поселений. Актуализированная редакция СНиП 2.07.01-89\"СВОД ПРАВИЛ ИНЖЕНЕРНЫЕ ИЗЫСКАНИЯ ДЛЯ СТРОИТЕЛЬСТВА ОСНОВНЫЕ ПОЛОЖЕНИЯ АКТУАЛИЗИРОВАННАЯ РЕДАКЦИЯ СНиП 11-02-96 Engineering survey for construction. Basic principles СП 47.13330.2016\""
                      "инструкция: Ответь на вопрос на основе документа."
                      "Если он не подходит, скажи об этом. Если в тексте не было вопроса или просьбы, попроси уточнить запрос."
                      "Отвечай вежливо. Отвечай только на русском языке. Ни в коем случае не используй иероглифы."
                      "Если с тобой здороваются, здоровайся в ответ. Если тебя спрашивают, что ты умеешь делать,"
                      "отвечай, что ты анализируешь документы и отвечаешь на вопросы по ним.\n"
                      f"НАЧАЛО ДОКУМЕНТА | {context} | КОНЕЦ ДОКУМЕНТА",
            "options": {
                "temperature": 0.2
            }
        }
        print(context)
        headers = {
            "Content-Type": "application/json"
        }
        return headers, data

    async def generate_simple_query_data(self, prompt: str) -> tuple[dict, dict]:
        data = {
            "model": self.model_name,
            "prompt": f"{prompt}",
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json"
        }
        return headers, data

    async def generate_table_description(self, table_data: str, num_questions: int) -> list[str] | None:

        prompt = f"""
                  Придумай {num_questions} вопросов к следующей таблице. Каждый вопрос в ответе должен начинаться с новой строчки:
                  {table_data}
                  """

        headers, data = await self.generate_simple_query_data(prompt)
        questions = await self.generate_response(headers, data)
        questions = questions.split("\n")
        return questions