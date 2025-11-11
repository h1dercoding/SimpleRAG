# -*- coding: UTF-8 -*-
'''
RAG知识库检索系统Flask后端API - 支持流式响应
'''

import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from file_loaders import load_specific_documents, load_documents_from_directory, split_documents, format_source_documents
from file_tracking import check_files_changed
from vector_store import initialize_or_load_vector_store, create_retriever
from qa_chain import create_qa_chain_with_memory
from flask_cors import CORS  # 用于处理跨域问题
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory

# 知识库文件夹路径
KNOWLEDGE_DIR = "./knowledge_base"
VECTOR_STORE_DIR = "./vector_store"
FILE_TRACKING_JSON = "./vector_store/file_tracking.json"
EMBEDDING_MODEL = "bge-m3:latest"  # Ollama中的嵌入模型
LLM_MODEL = "qwen2.5:7b"  # LLM模型名称

# 确保目录存在
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS，允许跨域请求

# 全局变量存储
qa_chain = None
vector_store = None
retriever = None
llm = None
prompt = None
chat_history_store = {}  # 存储不同会话的聊天历史

def initialize_rag_system():
    """初始化RAG系统"""
    global qa_chain, vector_store, retriever, llm, prompt
    
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
    
    # 4. 初始化LLM
    llm = ChatOllama(model=LLM_MODEL, streaming=True)
    
    # 5. 创建提示模板
    prompt_template = """使用以下上下文和对话历史来回答用户的问题。
    如果你不确定答案，请基于上下文信息提供你认为最合理的回答，不要编造不在上下文中的信息。

    上下文信息:
    {context}
    
    对话历史:
    {chat_history}
    
    用户问题: {question}
    
    请提供详细回答:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # 6. 创建标准问答链(非流式)用于其他功能
    qa_chain = create_qa_chain_with_memory(retriever, llm_model=LLM_MODEL)
    
    return qa_chain is not None

@app.route('/api/files', methods=['GET'])
def get_knowledge_files():
    """获取知识库文件列表"""
    knowledge_files = []
    if os.path.exists(KNOWLEDGE_DIR):
        for root, _, files in os.walk(KNOWLEDGE_DIR):
            for file in files:
                if not file.startswith('.'):  # 忽略隐藏文件
                    rel_path = os.path.relpath(os.path.join(root, file), KNOWLEDGE_DIR)
                    knowledge_files.append({
                        "name": rel_path,
                        "size": os.path.getsize(os.path.join(root, file)) / 1024,  # KB
                        "type": os.path.splitext(file)[1][1:].upper()
                    })
    
    return jsonify({
        "status": "success",
        "files": knowledge_files
    })

@app.route('/api/query', methods=['POST'])
def query():
    """处理问题查询请求 - 非流式响应"""
    global qa_chain
    
    if not qa_chain:
        if not initialize_rag_system():
            return jsonify({
                'status': 'error',
                'message': 'RAG系统初始化失败'
            })
    
    # 从请求中获取问题和会话ID
    data = request.json
    question = data.get('question', '')
    session_id = data.get('session_id', 'default')
    
    if not question:
        return jsonify({
            'status': 'error',
            'message': '问题不能为空'
        })
        
    try:
        # 获取当前会话的聊天历史
        if session_id not in chat_history_store:
            chat_history_store[session_id] = []
        
        # 保存用户问题到历史
        chat_history_store[session_id].append({"role": "user", "content": question})
        
        # 执行查询
        result = qa_chain.invoke({"question": question})
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        # 保存系统回答到历史
        chat_history_store[session_id].append({"role": "assistant", "content": answer})
        
        # 限制历史记录长度
        if len(chat_history_store[session_id]) > 20:
            chat_history_store[session_id] = chat_history_store[session_id][-20:]
        
        # 格式化知识来源
        sources = []
        for i, doc in enumerate(source_docs, 1):
            source_file = doc.metadata.get("source_file", "未知文件")
            source_info = {"file": source_file}
            if "page" in doc.metadata:
                source_info["page"] = doc.metadata["page"]
            sources.append(source_info)
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'sources': sources
        })
        
    except Exception as e:
        # 如果查询出错，从历史中移除最后的问题
        if session_id in chat_history_store and chat_history_store[session_id]:
            chat_history_store[session_id].pop()
        
        return jsonify({
            'status': 'error',
            'message': f'查询处理错误: {str(e)}'
        })

@app.route('/api/stream', methods=['POST'])
def stream_query():
    """处理问题查询请求 - 流式响应"""
    global retriever, llm, prompt
    
    if not retriever or not llm or not prompt:
        if not initialize_rag_system():
            return jsonify({
                'status': 'error',
                'message': 'RAG系统初始化失败'
            })
    
    # 从请求中获取问题和会话ID
    data = request.json
    question = data.get('question', '')
    session_id = data.get('session_id', 'default')
    
    if not question:
        return jsonify({
            'status': 'error',
            'message': '问题不能为空'
        })
    
    # 获取对话历史
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    
    # 添加用户问题到历史
    chat_history_store[session_id].append({"role": "user", "content": question})
    
    try:
        # 准备上下文和历史
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        history = chat_history_store[session_id][:-1]  # 不包括当前问题
        history_text = ""
        
        if len(history) > 0:
            for i in range(0, len(history), 2):
                if i+1 < len(history):
                    history_text += f"用户: {history[i]['content']}\n"
                    history_text += f"助手: {history[i+1]['content']}\n\n"
        
        # 创建流式响应
        def generate():
            # 发送源文档信息
            sources = []
            for doc in docs:
                source_file = doc.metadata.get("source_file", "未知文件")
                source_info = {"file": source_file}
                if "page" in doc.metadata:
                    source_info["page"] = doc.metadata["page"]
                sources.append(source_info)
            
            # 发送data前缀，确保SSE格式正确
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # 创建流式生成的链
            chain = (
                {"context": lambda _: context, "question": lambda _: question, "chat_history": lambda _: history_text}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # 流式生成回答
            answer = ""
            for chunk in chain.stream({}):
                answer += chunk
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
            
            # 保存回答到历史
            chat_history_store[session_id].append({"role": "assistant", "content": answer})
            
            # 限制历史记录长度
            if len(chat_history_store[session_id]) > 20:
                chat_history_store[session_id] = chat_history_store[session_id][-20:]
                
            # 流式响应结束
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        })

    except Exception as e:
        # 如果出错，从历史中移除最后的问题
        if chat_history_store[session_id]:
            chat_history_store[session_id].pop()
        return jsonify({
            'status': 'error',
            'message': f'流式查询处理错误: {str(e)}'
        })

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """获取指定会话的聊天历史"""
    history = chat_history_store.get(session_id, [])
    return jsonify({
        'status': 'success',
        'history': history
    })

@app.route('/api/history/<session_id>', methods=['DELETE'])
def clear_history(session_id):
    """清除指定会话的聊天历史"""
    chat_history_store[session_id] = []
    return jsonify({
        'status': 'success',
        'message': '聊天历史已清除'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件到知识库"""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '没有文件被上传'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': '未选择文件'
        })
    
    if file:
        # 确保文件名安全
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        
        # 保存文件
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        file.save(file_path)
        
        # 重新初始化RAG系统，处理新文件
        global qa_chain, vector_store
        qa_chain = None  # 强制重新初始化
        initialize_rag_system()
        
        return jsonify({
            'status': 'success',
            'message': f'文件 {filename} 上传成功并已加入知识库'
        })

@app.route('/api/files/<path:filename>', methods=['GET'])
def download_file(filename):
    """下载知识库中的文件"""
    return send_from_directory(KNOWLEDGE_DIR, filename, as_attachment=True)

if __name__ == '__main__':
    initialize_rag_system()
    app.run(debug=True, host='0.0.0.0', port=8080)