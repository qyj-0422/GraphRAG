# AdaptedGraphRAG.py
from Core.GraphRAG import GraphRAG
from Core.Graph.ExistingGraph import ExistingGraph  # 新增已有图加载类
from typing import Union, Any
from Core.Common.Logger import logger
import tiktoken
from pydantic import BaseModel, model_validator
from Core.Schema.RetrieverContext import RetrieverContext
from Core.Common.TimeStatistic import TimeStatistic
from Core.Storage.NameSpace import Workspace

class AdaptedGraphRAG(GraphRAG):
    """继承GraphRAG，支持直接加载已有图结构数据"""

    def __init__(self, config):
        super().__init__(config)

    @model_validator(mode="after")
    def _update_context(cls, data):
        if True:
            # 使用已有图数据的初始化逻辑
            cls.ENCODER = tiktoken.encoding_for_model(data.config.token_model)
            cls.workspace = Workspace(data.config.working_dir, data.config.index_name)
            cls.graph = ExistingGraph(data.config, llm=data.llm, encoder=cls.ENCODER)
            cls.doc_chunk = None  # 禁用分块
            # 初始化其他必要组件
            cls.time_manager = TimeStatistic()
            cls.retriever_context = RetrieverContext()
            data = cls._init_storage_namespace(data)
            data = cls._register_vdbs(data)
            data = cls._register_community(data)
            data = cls._register_e2r_r2c_matrix(data)
            data = cls._register_retriever_context(data)
            return data
        else:
            # 保持原有逻辑
            return super()._update_context(data)

    async def insert(self, docs: Union[str, list[Any]]):
        await super()._build_retriever_context()
        if True:
            logger.info("使用已有图数据，跳过文本分块和图构建阶段")
            return  # 直接返回，不执行构建逻辑
        else:
            # 保持原有逻辑
            logger.info("use_existing_graph参数有误")
            return
