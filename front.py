# -*- coding: UTF-8 -*-
'''
RAG知识库检索系统 - Streamlit前端界面 - 支持流式输出
'''

import os
import json
import time
import uuid
import requests
import streamlit as st
import pandas as pd
from io import BytesIO
from sseclient import SSEClient

# API服务地址
API_BASE_URL = "http://localhost:8080/api"

# 设置页面配置
st.set_page_config(
    page_title="RAG知识库检索系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2rem !important;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: bold;
        color: #2563EB;
    }
    .source-info {
        background-color: #F3F4F6;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .source-item {
        display: inline-block;
        background-color: #E5E7EB;
        border-radius: 12px;
        padding: 3px 10px;
        margin-right: 8px;
        margin-bottom: 5px;
        font-size: 0.8rem;
    }
    .stChatMessage {
        padding: 20px 20px 20px 20px !important;
    }
    .user-message {
        background-color: #DBEAFE !important;
    }
    .assistant-message {
        background-color: #F8FAFC !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'files' not in st.session_state:
    st.session_state.files = []

# 侧边栏
with st.sidebar:
    st.markdown('<div class="main-header">📚 RAG知识库检索系统</div>', unsafe_allow_html=True)
    
    # 文件上传部分
    st.markdown('<div class="sub-header">📤 上传文档</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("选择要上传到知识库的文件", 
                                   type=["pdf", "txt", "csv", "xlsx", "xls", "md"],
                                   help="支持PDF, TXT, CSV, Excel和Markdown格式")
    
    if uploaded_file is not None:
        with st.spinner("文件上传中..."):
            # 将文件发送到API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                result = response.json()
                
                if result.get("status") == "success":
                    st.success(result.get("message", "文件上传成功"))
                    # 刷新文件列表
                    st.session_state.files = []
                    time.sleep(1)  # 给后端一点时间处理文件
                else:
                    st.error(result.get("message", "文件上传失败"))
            except Exception as e:
                st.error(f"上传出错: {str(e)}")
    
    # 知识库文件列表
    st.markdown('<div class="sub-header">📑 知识库文件</div>', unsafe_allow_html=True)
    
    # 刷新按钮
    if st.button("刷新文件列表"):
        st.session_state.files = []
    
    # 加载文件列表
    if not st.session_state.files:
        with st.spinner("加载文件列表..."):
            try:
                response = requests.get(f"{API_BASE_URL}/files")
                result = response.json()
                
                if result.get("status") == "success":
                    st.session_state.files = result.get("files", [])
                else:
                    st.error("获取文件列表失败")
            except Exception as e:
                st.error(f"API连接错误: {str(e)}")
    
    # 显示文件列表
    if st.session_state.files:
        file_data = []
        for file in st.session_state.files:
            file_data.append({
                "文件名": file["name"],
                "大小(KB)": round(file["size"], 2),
                "类型": file["type"]
            })
        
        # 创建DataFrame显示文件列表
        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("知识库中暂无文件")
    
    # 清除聊天历史
    st.markdown('<div class="sub-header">🧹 聊天管理</div>', unsafe_allow_html=True)
    if st.button("清除对话历史"):
        try:
            response = requests.delete(f"{API_BASE_URL}/history/{st.session_state.session_id}")
            result = response.json()
            
            if result.get("status") == "success":
                st.session_state.chat_history = []
                st.success("对话历史已清除")
            else:
                st.error("清除对话历史失败")
        except Exception as e:
            st.error(f"API连接错误: {str(e)}")
    
    # 系统信息
    st.markdown('<div class="sub-header">ℹ️ 系统信息</div>', unsafe_allow_html=True)
    st.info(f"会话ID: {st.session_state.session_id[:8]}...")
    st.info(f"知识库文件数: {len(st.session_state.files)}")
    st.info("嵌入模型: bge-m3:latest")
    st.info("LLM模型: qwen2.5:7b")
    
    # 底部版权信息
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<center>© 2025 RAG知识库检索系统</center>", unsafe_allow_html=True)

# 主界面
st.markdown('<div class="main-header">🤖 RAG知识库问答助手</div>', unsafe_allow_html=True)

# 加载历史对话
if not st.session_state.chat_history:
    try:
        response = requests.get(f"{API_BASE_URL}/history/{st.session_state.session_id}")
        result = response.json()
        
        if result.get("status") == "success":
            st.session_state.chat_history = result.get("history", [])
    except Exception:
        # 如果API不可用，使用空历史记录
        pass

# 显示聊天界面
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        if role == "assistant" and "sources" in message:
            st.write(content)
            
            # 显示知识来源
            sources = message.get("sources", [])
            if sources:
                source_html = '<div class="source-info"><strong>知识来源:</strong><div>'
                for source in sources:
                    source_text = source["file"]
                    if "page" in source:
                        source_text += f" (第{source['page']}页)"
                    source_html += f'<span class="source-item">{source_text}</span>'
                source_html += '</div></div>'
                st.markdown(source_html, unsafe_allow_html=True)
        else:
            st.write(content)

# 聊天输入
user_input = st.chat_input("输入您的问题...")

if user_input:
    # 显示用户输入
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)
    
    # 将用户消息添加到历史记录
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 显示思考状态
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("思考中...")
        
        try:
            # 准备发送流式请求
            full_response = ""
            sources = []
            
            # 注释掉SSEClient方式，改用原生requests流式处理
            response = requests.post(
                f"{API_BASE_URL}/stream",
                json={"question": user_input, "session_id": st.session_state.session_id},
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            # 使用更简单直接的方式处理SSE流
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode('utf-8')
                
                # 仅处理SSE数据行
                if line.startswith('data: '):
                    # 去除前缀'data: '
                    event_data = line[6:]
                    try:
                        data = json.loads(event_data)
                        event_type = data.get("type")
                        
                        if event_type == "token":
                            # 添加新的token到响应中
                            token = data.get("token", "")
                            full_response += token
                            
                            # 更新界面
                            message_placeholder.markdown(full_response)
                            
                        elif event_type == "sources":
                            # 保存知识来源信息
                            sources = data.get("sources", [])
                            
                        elif event_type == "end":
                            # 流结束，显示完整响应
                            if sources:
                                source_html = '<div class="source-info"><strong>知识来源:</strong><div>'
                                for source in sources:
                                    source_text = source["file"]
                                    if "page" in source:
                                        source_text += f" (第{source['page']}页)"
                                    source_html += f'<span class="source-item">{source_text}</span>'
                                source_html += '</div></div>'
                                
                                # 显示最终结果及来源
                                message_placeholder.markdown(f"{full_response}\n\n{source_html}", unsafe_allow_html=True)
                            
                            # 保存到历史记录
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": full_response,
                                "sources": sources
                            })
                            break
                    except json.JSONDecodeError:
                        # 忽略无效的JSON数据
                        continue
                    
        except Exception as e:
            message_placeholder.error(f"API连接错误: {str(e)}")

# 页脚说明
st.markdown("---")
st.markdown(
    """
    <small>
    使用说明: 
    1. 在左侧上传知识文档(PDF、TXT、Excel等)
    2. 在上方输入框中提问
    3. 系统将在知识库中检索相关内容并逐字显示回答
    4. 回答会显示知识来源
    </small>
    """, 
    unsafe_allow_html=True
)