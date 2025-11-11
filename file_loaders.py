# -*- coding: UTF-8 -*-
'''
文件加载模块，负责各种类型文件的加载和处理
'''

import os
import glob
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_file_loader(file_path: str):
    """根据文件类型获取适合的加载器"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension == ".txt":
        return TextLoader(file_path)
    elif file_extension == ".csv":
        return CSVLoader(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path)
    elif file_extension in [".md", ".markdown"]:
        return UnstructuredMarkdownLoader(file_path)
    else:
        # 对于不支持的文件类型，返回None
        print(f"不支持的文件类型: {file_path}")
        return None

def load_specific_documents(file_paths: List[str]) -> List[Document]:
    """加载指定文件列表中的文档"""
    documents = []
    
    for file_path in file_paths:
        try:
            print(f"正在加载: {file_path}")
            loader = get_file_loader(file_path)
            if loader:
                file_docs = loader.load()
                # 添加文件路径元数据
                for doc in file_docs:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["source_file"] = os.path.basename(file_path)
                    doc.metadata["file_path"] = file_path
                documents.extend(file_docs)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {str(e)}")
    
    return documents

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """从指定目录加载所有支持的文档"""
    supported_extensions = [".pdf", ".txt", ".csv", ".xlsx", ".xls", ".md", ".markdown"]
    documents = []
    
    # 获取所有文件路径
    all_files = []
    for ext in supported_extensions:
        all_files.extend(glob.glob(os.path.join(directory_path, f"**/*{ext}"), recursive=True))
    
    print(f"发现 {len(all_files)} 个可加载文件")
    
    # 加载每个文件
    for file_path in all_files:
        try:
            print(f"正在加载: {file_path}")
            loader = get_file_loader(file_path)
            if loader:
                file_docs = loader.load()
                # 添加文件路径元数据
                for doc in file_docs:
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["source_file"] = os.path.basename(file_path)
                    doc.metadata["file_path"] = file_path
                documents.extend(file_docs)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {str(e)}")
    
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """将文档拆分为更小的文本块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)

def format_source_documents(source_documents: List[Document]) -> str:
    """格式化源文档信息，用于展示知识溯源"""
    sources = []
    for i, doc in enumerate(source_documents, 1):
        source_file = doc.metadata.get("source_file", "未知文件")
        source = f"[{i}] {source_file}"
        if "page" in doc.metadata:
            source += f" (第 {doc.metadata['page']} 页)"
        sources.append(source)
    
    return "\n".join(sources)