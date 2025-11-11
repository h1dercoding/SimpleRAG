# -*- coding: UTF-8 -*-
'''
问答链模块，负责问答系统和对话历史管理
'''

from typing import List
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

def create_qa_chain_with_memory(retriever, llm_model: str = "qwen2.5:7b"):
    """创建带有记忆功能的问答链"""
    # 初始化LLM
    llm = ChatOllama(model=llm_model)
    
    # 创建对话记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # 自定义提示模板，包含对话历史
    prompt_template = """使用以下上下文和对话历史来回答用户的问题。
    如果你不确定答案，请基于上下文信息提供你认为最合理的回答，不要编造不在上下文中的信息。

    上下文信息:
    {context}
    
    对话历史:
    {chat_history}
    
    用户问题: {question}
    
    请提供详细回答:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # 创建会话检索链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain
    
def display_chat_history(chat_history):
    """显示聊天历史记录"""
    print("\n=== 对话历史 ===")
    for i, message in enumerate(chat_history):
        role = "用户" if i % 2 == 0 else "系统"
        print(f"{role}: {message}")