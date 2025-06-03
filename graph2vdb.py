#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graph2vdb.py - Convert NetworkX graph to Vector Database
Directly build VDB from existing NetworkX graph object
"""

import asyncio
import networkx as nx
from typing import Optional, Dict, Any
from pathlib import Path

from Core.GraphRAG import GraphRAG
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Storage.NameSpace import Workspace
from Core.Index import get_index, get_index_config
from Option.Config2 import Config
from pydantic import model_validator
import tiktoken
from Core.Common.TimeStatistic import TimeStatistic
from Core.Schema.RetrieverContext import RetrieverContext
from pydantic import ConfigDict


class DirectNetworkXStorage(NetworkXStorage):
    """
    Extended NetworkXStorage that can directly load from NetworkX graph object
    """

    def __init__(self, graph: Optional[nx.Graph] = None):
        super().__init__()
        if graph is not None:
            self._graph = graph
            logger.info(
                f"Loaded graph with {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges")

    async def load_graph(self, force: bool = False) -> bool:
        """Override to prevent loading from file when graph is already set"""
        if self._graph is not None and not force:
            return True
        return await super().load_graph(force)


class DirectGraph(BaseGraph):
    """
    Graph class that can work with pre-existing NetworkX graph
    """

    def __init__(self, config, llm, encoder, nx_graph: Optional[nx.Graph] = None):
        super().__init__(config, llm, encoder)
        self._graph = DirectNetworkXStorage(nx_graph)

    async def build_graph(self, chunks=None, force: bool = False):
        """Override to skip graph building when graph is pre-loaded"""
        logger.info("Using pre-existing graph, skipping build phase")
        is_exist = await self._load_graph(force)
        if not is_exist:
            raise ValueError("No graph loaded!")
        logger.info("✅ Graph ready for VDB construction")

    async def _build_graph(self, chunks):
        """Not needed for direct graph loading"""
        pass

    async def _extract_entity_relationship(self, chunk_key_pair):
        """Not needed for direct graph loading"""
        pass


class DirectGraphRAG(GraphRAG):
    """
    Modified GraphRAG that works with pre-existing NetworkX graphs
    """
    nx_graph: Optional[nx.Graph] = None  # 可选字段，默认值为 None

    def __init__(self, config, nx_graph: Optional[nx.Graph] = None):
        super().__init__(config=config)
        self.nx_graph = nx_graph

    @model_validator(mode="after")
    def _update_context(cls, data):
        """Override to use DirectGraph instead of regular graph"""
        # Set up basic components first
        cls.ENCODER = tiktoken.encoding_for_model(data.config.token_model)
        cls.workspace = Workspace(data.config.working_dir, data.config.exp_name)
        cls.time_manager = TimeStatistic()  # Use existing time manager
        cls.retriever_context = RetrieverContext()  # Use existing retriever context

        # Create DirectGraph with the pre-loaded NetworkX graph
        cls.graph = DirectGraph(
            data.config.graph,
            llm=data.llm,
            encoder=cls.ENCODER,
            nx_graph=data.nx_graph
        )

        # Set up doc_chunk (minimal setup as we're not using it for graph construction)
        cls.doc_chunk = None

        # Update storage namespaces
        data.graph.namespace = data.workspace.make_for("graph_storage")

        # Register other components following parent logic
        data = cls._init_storage_namespace(data)
        data = cls._register_vdbs(data)
        data = cls._register_community(data)
        data = cls._register_e2r_r2c_matrix(data)
        data = cls._register_retriever_context(data)

        return data

    async def build_vdb_from_graph(self):
        """
        Build VDB from the loaded graph without text processing
        """
        logger.info("Building VDB from existing graph...")

        # Ensure graph is loaded
        await self.graph.build_graph(force=False)

        # Build entity VDB if enabled
        if self.config.use_entities_vdb:
            node_metadata = await self.graph.node_metadata()
            if not node_metadata:
                logger.warning("No node metadata found. Skipping entity indexing.")
            else:
                await self.entities_vdb.build_index(
                    await self.graph.nodes_data(),
                    node_metadata,
                    force=False
                )
                logger.info("✅ Entity VDB built successfully")

        # Build relation VDB if enabled
        if self.config.use_relations_vdb:
            edge_metadata = await self.graph.edge_metadata()
            if not edge_metadata:
                logger.warning("No edge metadata found. Skipping relation indexing.")
            else:
                await self.relations_vdb.build_index(
                    await self.graph.edges_data(),
                    edge_metadata,
                    force=False
                )
                logger.info("✅ Relation VDB built successfully")

        # Build subgraph VDB if enabled
        if self.config.use_subgraphs_vdb:
            subgraph_metadata = await self.graph.subgraph_metadata()
            if not subgraph_metadata:
                logger.warning("No subgraph metadata found. Skipping subgraph indexing.")
            else:
                await self.subgraphs_vdb.build_index(
                    await self.graph.subgraphs_data(),
                    subgraph_metadata,
                    force=False
                )
                logger.info("✅ Subgraph VDB built successfully")

        # Build community if enabled
        if self.config.graph.use_community:
            await self.community.cluster(
                largest_cc=await self.graph.stable_largest_cc(),
                max_cluster_size=self.config.graph.max_graph_cluster_size,
                random_seed=self.config.graph.graph_cluster_seed,
                force=False
            )
            await self.community.generate_community_report(self.graph, force=False)
            logger.info("✅ Community analysis completed")

        # Build retriever context
        await self._build_retriever_context()
        logger.info("✅ VDB construction completed")


async def graph2vdb(
        nx_graph: nx.Graph,
        config_path: Optional[Path] = None,
        config: Optional[Config] = None,
        dataset_name: str = "direct_graph",
        use_entities_vdb: Optional[bool] = None,
        use_relations_vdb: Optional[bool] = None,
        use_subgraphs_vdb: Optional[bool] = None,
        vdb_type: Optional[str] = None,
        working_dir: Optional[str] = None,
        exp_name: Optional[str] = None
) -> DirectGraphRAG:
    """
    Convert NetworkX graph to Vector Database

    Args:
        nx_graph: NetworkX graph object
        config_path: Path to config YAML file (optional)
        config: Config object (optional, if not using config_path)
        dataset_name: Dataset name for config parsing
        use_entities_vdb: Whether to build entity VDB (None to use config default)
        use_relations_vdb: Whether to build relation VDB (None to use config default)
        use_subgraphs_vdb: Whether to build subgraph VDB (None to use config default)
        vdb_type: Type of VDB ("vector", "faiss", "colbert") (None to use config default)
        working_dir: Output directory (None to use config default)
        exp_name: Experiment name (None to use config default)

    Returns:
        DirectGraphRAG instance with built VDBs
    """

    # Load or create config
    if config is None:
        if config_path:
            # Use the parse method which properly merges with defaults
            config = Config.parse(Path(config_path), dataset_name)
        else:
            # Use default config
            config = Config.default()
            config.dataset_name = dataset_name

    # Override config with function parameters if provided
    if use_entities_vdb is not None:
        config.use_entities_vdb = use_entities_vdb
    if use_relations_vdb is not None:
        config.use_relations_vdb = use_relations_vdb
    if use_subgraphs_vdb is not None:
        config.use_subgraphs_vdb = use_subgraphs_vdb
    if vdb_type is not None:
        config.vdb_type = vdb_type
    if working_dir is not None:
        config.working_dir = working_dir
    if exp_name is not None:
        config.exp_name = exp_name

    # Create DirectGraphRAG instance
    graph_rag = DirectGraphRAG(config=config, nx_graph=nx_graph)

    # Build VDBs
    await graph_rag.build_vdb_from_graph()

    return graph_rag


def graph2vdb_sync(
        nx_graph: nx.Graph,
        **kwargs
) -> DirectGraphRAG:
    """
    Synchronous wrapper for graph2vdb
    """
    return asyncio.run(graph2vdb(nx_graph, **kwargs))


# Example usage
if __name__ == "__main__":
    import os

    # Example 1: Load existing graph from file
    graph_file = "./yago/er_graph/nx_data.graphml"
    if os.path.exists(graph_file):

        G = nx.read_graphml(graph_file)
        logger.info(f'existing graph read successfully，graph have {len(G)} nodes')
    else:
        # Example 2: Create a sample graph
        logger.error('not successfully read graph')
        G = nx.Graph()

        # Add nodes with attributes (following GraphRAG schema)
        G.add_node("entity1",
                   entity_name="Entity 1",
                   entity_type="Person",
                   description="First entity description",
                   source_id="doc1")
        G.add_node("entity2",
                   entity_name="Entity 2",
                   entity_type="Organization",
                   description="Second entity description",
                   source_id="doc1")
        G.add_node("entity3",
                   entity_name="Entity 3",
                   entity_type="Location",
                   description="Third entity description",
                   source_id="doc2")

        # Add edges with attributes
        G.add_edge("entity1", "entity2",
                   relation_name="works_for",
                   description="Entity 1 works for Entity 2",
                   weight=1.0,
                   source_id="doc1",
                   src_id="entity1",
                   tgt_id="entity2")
        G.add_edge("entity2", "entity3",
                   relation_name="located_in",
                   description="Entity 2 is located in Entity 3",
                   weight=0.8,
                   source_id="doc2",
                   src_id="entity2",
                   tgt_id="entity3")

    # Method 1: Use with config file
    graph_rag = graph2vdb_sync(
        nx_graph=G,
        config_path=Path("Option/Method/LightRAG.yaml"),
        dataset_name="yago",
        use_entities_vdb=True,
        use_relations_vdb=True,
        working_dir="./graph_vdb_output",
        exp_name="my_graph_vdb"
    )


    # Method 2: Use with default config
    # graph_rag = graph2vdb_sync(
    #     G,
    #     dataset_name="my_graph_dataset",
    #     use_entities_vdb=True,
    #     use_relations_vdb=False,
    #     vdb_type="vector"
    # )

    # Now you can use the query functionality
    # async def test_query():
    #     response = await graph_rag.query("What is Entity 1?")
    #     print(f"Query response: {response}")
    #
    #
    # asyncio.run(test_query())