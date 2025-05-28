import io

import pandas as pd
from tqdm import tqdm

from docx import Document
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from fastapi import HTTPException
from loguru import logger

from src.common.config.config import Config
from src.llm.llm_service import LlmService
from src.vectorizer.vectorizer_service import VectorizerService


class ElasticService:
    def __init__(self, config: Config, vectorizer_service: VectorizerService, llm_service: LlmService):
        self.client = Elasticsearch(hosts=[f"http://{config.get('ELASTIC_HOST')}:{config.get('ELASTIC_PORT')}"])
        self.config = config
        self.vectorizer_service = vectorizer_service
        self.llm_service = llm_service

    async def delete_documents_from_index(self, index_name: str) -> str:
        try:
            self.client.delete_by_query(index=index_name, body={"query": {"match_all": {}}})
            return f"Successfully deleted all documents from index {index_name}"
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail=e.__str__())

    async def search(self, embedding: list) -> ObjectApiResponse:
        index_name = self.config.get("ELASTIC_DOCUMENT_INDEX")
        query_body = {
            "knn": {
                "field": "body_vector",
                "query_vector": embedding,
                "k": 20,
                "num_candidates": 20
            },
            "_source": ["body"],
        }
        return self.client.search(index=index_name, body=query_body)

    async def get_last_index(self, index_name: str) -> int:
        query_body = {
            "size": 1,
            "sort": [
                {
                    "num_id": {
                        "order": "desc"
                    }
                }
            ]
        }
        try:
            last_id_data = self.client.search(index=index_name, body=query_body)
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=500, detail=e.__str__())
        if last_id_data.body["hits"]["hits"]:
            return last_id_data.body["hits"]["hits"][0]["_source"]["num_id"]
        return 0

    async def upload_to_index(self, file: bytes, index_name: str):
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body={
                "mappings": {
                    "properties": {
                        "body_vector": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "body": {
                            "type": "text"
                        },
                        "num_id": {
                            "type": "long"
                        }
                    }
                }})
        documents = []
        full_doc = Document(io.BytesIO(file))
        ids = await self.get_last_index(index_name)
        logger.info(f"Started uploading documents to index {index_name} from id {ids}")
        for index, paragraph in tqdm(enumerate(full_doc.paragraphs), total=len(full_doc.paragraphs), desc="Processing texts"):
            if paragraph.text.rstrip() == "":
                continue
            # Create sentence embedding
            vector = self.encode(paragraph.text)
            doc = {
                "_id": str(index),
                "num_id": index,
                "body": paragraph.text,
                "body_vector": vector,
            }
            # Append JSON document to a list.
            documents.append(doc)
            ids += 1
        # for index, table in tqdm(enumerate(full_doc.tables), total=len(full_doc.tables) ,desc="Processint tables"):
        #     table_data = []
        #     for row in table.rows:
        #         row_data = [cell.text.strip() for cell in row.cells]
        #         table_data.append(row_data)
        #     table_description = await self.llm_service.generate_table_description(table_data)
        #     vector = self.encode(table_description)
        #     doc = {
        #         "_id": str(ids + index),
        #         "num_id": index,
        #         "body": str(table_description),
        #         "body_vector": vector,
        #     }
        #     documents.append(doc)

        for table in tqdm(full_doc.tables, total=len(full_doc.tables) ,desc="Processint tables"):
            last_index = 0
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            table_df = pd.DataFrame(table_data[1:], columns=table_data[0])
            for index, row in table_df.iterrows():
                # table_data.append(row_data)
                # table_description = await self.llm_service.generate_table_description(table_data)
                vector = self.encode(str(row))
                doc = {
                    "_id": str(ids + index + 1),
                    "num_id": index,
                    "body": str(row),
                    "body_vector": vector,
                }
                documents.append(doc)
                last_index += 1
            ids += last_index

        if documents:
            bulk(self.client, documents, index=index_name)
        print("Finished")
        return index_name

    def encode(self, document: str) -> list:
        try:
            return self.vectorizer_service.embed(document)
        except Exception as e:
            raise HTTPException(500, str(e))
