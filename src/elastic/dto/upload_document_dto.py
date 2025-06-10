from pydantic import BaseModel, Field, field_validator
from src.common.exceptions.http_exception import http_exception


class UploadDocumentDTO(BaseModel):

    index_name: str = Field(description="index name in elastic")
    table_context_size: int = Field(default=5, examples=[5], description="table context size in paragraphs")
    table_questions_num: int = Field(default=10, examples=[10],description="number of questions for table")

    @field_validator("index_name", mode="before")
    @classmethod
    def validate_index_name(cls, value: str):
        if value.startswith("_") or value.startswith("."):
            raise http_exception(
                400,
                "Index name must not start with an underscore or dot ('_', '.')",
                _input=value,
                _detail={"example": "index_name"},
            )
        return value
