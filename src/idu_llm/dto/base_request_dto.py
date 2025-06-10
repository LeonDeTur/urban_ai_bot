from pydantic import BaseModel, Field


class BaseLlmRequest(BaseModel):

    index_name: str = Field(..., examples=["prostor_doc"], description="ElasticSearch index name to use as context db")
    user_request: str = Field(..., examples=["Что ты умеешь?"])
