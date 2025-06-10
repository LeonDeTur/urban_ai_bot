from typing import Annotated

from IPython.terminal.shortcuts.auto_suggest import accept_word
from fastapi import APIRouter, UploadFile, Depends, Body, Query

from src.dependencies import elastic_client, config
from src.elastic.dto.elastic_search_dto import ElasticSearchDTO
from src.elastic.dto.upload_document_dto import UploadDocumentDTO

elastic_router = APIRouter()
tag = ["LLM Controller"]
cfg_tag = ["Config Controller"]


@elastic_router.get("/llm/indexes", tags=tag)
async def get_available_indexes():
    return await elastic_client.get_available_indexes()

@elastic_router.post("/llm/indexes", tags=tag)
async def create_index(index_name: str):
    return await elastic_client.create_index(index_name)

@elastic_router.post("/llm/upload_document", tags=tag)
async def upload_document(
        file: UploadFile,
        dto: Annotated[UploadDocumentDTO, Depends(UploadDocumentDTO)]
):
    return await elastic_client.upload_to_index(
        await file.read(),
        dto.index_name,
        dto.table_context_size,
        dto.table_questions_num,
    )

@elastic_router.delete("/llm/delete_documents/{index_name}", tags=tag)
async def delete_document(index_name: str):
    return await elastic_client.delete_documents_from_index(index_name)

@elastic_router.delete("/llm/delete_index/{index_name}", tags=tag)
async def delete_documents(index_name: str):
    return await elastic_client.delete_index(index_name)

@elastic_router.get("/llm/search", tags=tag)
async def search(
        dto: Annotated[ElasticSearchDTO, Depends(ElasticSearchDTO)]
):
    return await elastic_client.search(elastic_client.encode(dto.prompt))


@elastic_router.put("/cfg/configure", tags=cfg_tag)
async def configure(
        body: Annotated[dict, Body()],
):
    for k, v in body.items():
        config.set(k, v)


@elastic_router.get("/cfg", tags=cfg_tag)
async def get_env(key: Annotated[str, Query()]):
    return config.get(key)
