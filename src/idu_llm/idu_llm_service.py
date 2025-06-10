import json

import requests

from src.common.exceptions.http_exception import http_exception
from src.llm.llm_service import LlmService
from src.elastic.elastic_service import ElasticService
from src.vectorizer.vectorizer_service import VectorizerService
from .dto.base_request_dto import BaseLlmRequest


class IduLLMService:

    def __init__(
            self,
            llm_service: LlmService,
            elastic_client: ElasticService,
            vectorizer_model: VectorizerService,
    ):

        self.llm_service = llm_service
        self.elastic_client = elastic_client
        self.vectorizer_model = vectorizer_model

    async def generate_response(self, message_info: BaseLlmRequest) -> str:
        try:
            embedding = self.vectorizer_model.embed(message_info.user_request)
        except Exception as e:
            raise http_exception(
                500,
                "Error during creating embedding",
                _input=message_info.user_request,
                _detail=e.__str__(),
            )
        try:
            elastic_response = await self.elastic_client.search(embedding, message_info.index_name)
        except Exception as e:
            raise http_exception(
                500,
                "Error during creating extracting elastic document",
                _input={
                    "message_info.user_request": message_info.user_request,
                    "embedding": embedding,
                },
                _detail=e.__str__()
            )
        context = ';'.join([resp["_source"]["body"].rstrip() for resp in elastic_response["hits"]["hits"]])
        headers, data = await self.llm_service.generate_request_data(message_info.user_request, context, False)
        try:
            llm_response = requests.post(
                f"{self.llm_service.url}/api/generate",
                headers=headers,
                data=json.dumps(data),
            )
        except Exception as e:
            raise http_exception(
                500,
                "Error during retrieving response",
                _input={
                    "message_info.user_request": message_info.user_request,
                    "embedding": embedding,
                    "context": context,
                    "llm_request_headers": headers,
                    "formed_data": data,
                },
                _detail=e.__str__(),
            )
        if llm_response.status_code != 200:
            raise http_exception(
                llm_response.status_code,
                "Error during generating llm request",
                _input={
                    "message_info.user_request": message_info.user_request,
                    "embedding": embedding,
                    "context": context,
                    "llm_request_headers": headers,
                    "formed_data": data,
                },
                _detail=llm_response.text
            )
        return llm_response.json()

    async def generate_stream_response(self, message_info: BaseLlmRequest):
        try:
            embedding = self.vectorizer_model.embed(message_info.user_request)
        except Exception as e:
            raise http_exception(
                500,
                "Error during creating embedding",
                _input=message_info.user_request,
                _detail=e.__str__(),
            )
        try:
            elastic_response = await self.elastic_client.search(embedding, message_info.index_name)
        except Exception as e:
            raise http_exception(
                500,
                "Error during creating extracting elastic document",
                _input={
                    "message_info.user_request": message_info.user_request,
                    "embedding": embedding,
                },
                _detail=e.__str__()
            )
        context = ';'.join([resp["_source"]["body"].rstrip() for resp in elastic_response["hits"]["hits"]])
        headers, data = await self.llm_service.generate_request_data(message_info.user_request, context, True)
        with requests.post(
            f"{self.llm_service.url}/api/generate",
            headers=headers,
            data=json.dumps(data),
            stream=True
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=512*1024):
                    chunk = json.loads(chunk)
                    if not chunk["done"]:
                        yield chunk["response"]
                    else:
                        yield False
