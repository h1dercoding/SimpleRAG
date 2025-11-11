# -*- coding: UTF-8 -*-
'''
文件跟踪模块，用于检测知识库文件的变更
'''

import os
import glob
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # 读取文件块并更新哈希
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()

def get_file_metadata(file_path):
    """获取文件的元数据信息"""
    file_stats = os.stat(file_path)
    return {
        "size": file_stats.st_size,
        "modified_time": file_stats.st_mtime,
        "hash": calculate_file_hash(file_path),
        "last_processed": datetime.now().isoformat()
    }

def load_file_tracking(tracking_file_path):
    """加载文件跟踪JSON，如果不存在则创建一个新的"""
    if os.path.exists(tracking_file_path):
        try:
            with open(tracking_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取跟踪文件出错: {str(e)}，将创建新的跟踪文件")
    
    # 创建一个新的跟踪数据结构
    return {"files": {}, "last_update": datetime.now().isoformat()}

def save_file_tracking(tracking_data, tracking_file_path):
    """保存文件跟踪数据到JSON文件"""
    tracking_data["last_update"] = datetime.now().isoformat()
    with open(tracking_file_path, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, ensure_ascii=False, indent=2)

def check_files_changed(knowledge_dir, tracking_file_path):
    """检查知识库文件是否有变更，返回新增和变更的文件列表"""
    supported_extensions = [".pdf", ".txt", ".csv", ".xlsx", ".xls", ".md", ".markdown"]
    
    # 加载文件跟踪数据
    tracking_data = load_file_tracking(tracking_file_path)
    tracked_files = tracking_data.get("files", {})
    
    # 获取所有当前文件
    current_files = []
    for ext in supported_extensions:
        current_files.extend(glob.glob(os.path.join(knowledge_dir, f"**/*{ext}"), recursive=True))
    
    # 检查新增和变更的文件
    new_or_changed_files = []
    for file_path in current_files:
        rel_path = os.path.relpath(file_path, knowledge_dir)
        
        # 检查文件是否在跟踪列表中
        if rel_path not in tracked_files:
            print(f"发现新文件: {rel_path}")
            new_or_changed_files.append(file_path)
            # 添加到跟踪列表
            tracked_files[rel_path] = get_file_metadata(file_path)
        else:
            # 检查文件是否有变更
            current_hash = calculate_file_hash(file_path)
            if current_hash != tracked_files[rel_path]["hash"]:
                print(f"文件已更新: {rel_path}")
                new_or_changed_files.append(file_path)
                # 更新跟踪信息
                tracked_files[rel_path] = get_file_metadata(file_path)
    
    # 检查是否有文件被删除
    current_rel_paths = [os.path.relpath(f, knowledge_dir) for f in current_files]
    for rel_path in list(tracked_files.keys()):
        if rel_path not in current_rel_paths:
            print(f"文件已删除: {rel_path}")
            del tracked_files[rel_path]
    
    # 保存更新后的跟踪数据
    tracking_data["files"] = tracked_files
    save_file_tracking(tracking_data, tracking_file_path)
    
    return new_or_changed_files