from pydantic import BaseModel
from pydantic import field_validator

from Core.Common.Constants import CONFIG_ROOT, LLM_API_TIMEOUT, METAGPT_ROOT
class BaseVectorStorage(BaseModel):
    # embedding_func: EmbeddingFunc
    # meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError

    