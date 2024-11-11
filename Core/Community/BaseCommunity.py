from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from Core.Provider.BaseLLM import BaseLLM
from typing import Optional

class BaseCommunity(ABC, BaseModel): 
    """Base community class definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    enforce_sub_communities: bool = False
    llm: Optional[BaseLLM] = Field(default=None, exclude=True)


    @abstractmethod
    async def _clustering_(self):
        pass

    @abstractmethod
    async def _generate_community_report_(self):
        pass
    

    @abstractmethod
    async def _community_schema_(self):
        pass