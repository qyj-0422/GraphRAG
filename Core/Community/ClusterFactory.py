
from __future__ import annotations


from pydantic import BaseModel
from Core.Common.Logger import logger
from Core.Community.BaseCommunity import BaseCommunity

class CommunityRegistry(BaseModel):
    communities: dict = {}

    def register_community(
        self,
        name: str,
        community_object = None,
        verbose: bool = False,
    ):
        
        if self.has_community(name):
            return
        
        self.communities[name] = community_object
    
        if verbose:
            logger.info(f"Community type {name} registered")

    def has_community(self, name: str) -> BaseCommunity:
        return name in self.communities

    def get_community(self, name) -> BaseCommunity:
        return self.communities.get(name)



# Registry instance
COM_REGISTRY = CommunityRegistry()


def register_community(name):
    """register a community to registry"""

    def decorator(cls):
     
        COM_REGISTRY.register_community(
            name = name,
            community_object=cls,
        )
        return cls

    return decorator


def get_community_instance(name: str, **kwargs) -> BaseCommunity:
    """
    Get a community instance from the registry.

    Args:
        name (str): The name of the community.

    Returns:
        BaseCommunity: The community instance.
    """
    community_class: type[BaseCommunity] = COM_REGISTRY.get_community(name)
    return community_class(**kwargs)
