# -*- coding: UTF-8 -*-
'''
RAG知识库检索系统主程序入口
'''

import os
from file_loaders import load_specific_documents, load_documents_from_directory, split_documents, format_source_documents
from file_tracking import check_files_changed
from vector_store import initialize_or_load_vector_store, create_retriever
from qa_chain import create_qa_chain_with_memory, display_chat_history

# 知识库文件夹路径
KNOWLEDGE_DIR = "./knowledge_base"
VECTOR_STORE_DIR = "./vector_store"
FILE_TRACKING_JSON = "./vector_store/file_tracking.json"
EMBEDDING_MODEL = "bge-m3:latest"  # Ollama中的嵌入模型
LLM_MODEL = "qwen2.5:7b"  # LLM模型名称

# 确保目录存在
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

def main():
    print("=== RAG知识库问答系统初始化 ===")
    
    # 1. 检查文件变更
    print("\n检查知识库文件变更...")
    changed_files = check_files_changed(KNOWLEDGE_DIR, FILE_TRACKING_JSON)
    
    # 2. 根据文件变更情况决定是否需要更新向量库
    if changed_files:
        print(f"\n发现 {len(changed_files)} 个新增或变更文件，需要更新向量库")
        
        # 2.1 只加载变更的文件
        documents = load_specific_documents(changed_files)
        print(f"加载了 {len(documents)} 个文档")
        
        # 2.2 文档切片
        chunked_documents = split_documents(documents)
        print(f"文档切片后共有 {len(chunked_documents)} 个文本块")
        
        # 2.3 更新向量存储
        vector_store = initialize_or_load_vector_store(
            docs=chunked_documents, 
            vector_store_dir=VECTOR_STORE_DIR, 
            embedding_model=EMBEDDING_MODEL
        )
        print("向量库已更新")
    else:
        print("\n未发现文件变更，直接加载现有向量库")
        
        # 检查向量库是否存在
        if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
            # 加载现有向量库
            vector_store = initialize_or_load_vector_store(
                vector_store_dir=VECTOR_STORE_DIR,
                embedding_model=EMBEDDING_MODEL
            )
            print("成功加载现有向量库")
        else:
            # 如果向量库不存在，需要处理所有文件
            print("向量库不存在，将处理所有文件")
            documents = load_documents_from_directory(KNOWLEDGE_DIR)
            chunked_documents = split_documents(documents)
            vector_store = initialize_or_load_vector_store(
                docs=chunked_documents,
                vector_store_dir=VECTOR_STORE_DIR,
                embedding_model=EMBEDDING_MODEL
            )
    
    # 3. 创建检索器
    retriever = create_retriever(vector_store, k=4)
    
    # 4. 创建带有记忆功能的问答链
    qa_chain = create_qa_chain_with_memory(retriever, llm_model=LLM_MODEL)
    
    # 5. 交互式问答
    print("\n=== RAG知识库问答系统（带记忆功能）===")
    print("输入问题进行查询，输入'history'查看对话历史，输入'exit'退出")
    
    # 存储简化的对话历史，便于显示
    chat_history = []
    
    while True:
        query = input("\n问题: ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'history':
            display_chat_history(chat_history)
            continue
            
        try:
            # 保存用户问题到历史
            chat_history.append(query)
            
            # 执行查询
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "")
            source_docs = result.get("source_documents", [])
            
            # 保存系统回答到历史
            chat_history.append(answer)
            
            # 输出结果
            print("\n回答:")
            print(answer)
            
            # 输出知识溯源
            print("\n知识来源:")
            print(format_source_documents(source_docs))
            
        except Exception as e:
            # 如果查询出错，从历史中移除最后的问题
            if chat_history:
                chat_history.pop()
            print(f"查询出错: {str(e)}")

if __name__ == "__main__":
    main()