# -*- coding: UTF-8 -*-
'''
向量存储模块，负责文档向量化和存储
'''

import os
from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def initialize_or_load_vector_store(docs: List[Document] = None, 
                                   vector_store_dir: str = "./vector_store", 
                                   embedding_model: str = "bge-m3:latest"):
    """初始化向量存储或加载现有存储"""
    # 设置嵌入模型
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # 检查是否存在现有向量存储
    if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
        print(f"加载现有向量存储: {vector_store_dir}")
        # 加载现有数据库
        vector_store = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings
        )
        
        # 如果提供了新文档，添加到现有存储中
        if docs:
            print(f"向现有向量存储添加 {len(docs)} 个新文档块")
            vector_store.add_documents(docs)
            # Chroma新版本不再需要显式调用persist方法
    else:
        # 创建新的向量存储
        if not docs:
            raise ValueError("首次创建向量存储时必须提供文档")
        
        print(f"创建新的向量存储，包含 {len(docs)} 个文档块")
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=vector_store_dir
        )
    
    return vector_store

def create_retriever(vector_store, k: int = 4):
    """创建检索器"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}  # 检索前k个相似文档
    )