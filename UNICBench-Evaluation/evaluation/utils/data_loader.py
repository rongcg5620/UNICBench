#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 集成自new_data/load_data.py
支持图片计数任务数据加载，为评测系统提供数据接口
"""

import os 
import json
from pathlib import Path
from typing import List, Dict, Any

def get_all_json(tag_dir):
    """获取目录下所有_label.json标注文件"""
    all_json_list = []
    # 递归遍历目标文件夹内的所有以_label.json结尾的文件, 将符合条件的文件路径添加到列表中
    for root, dirs, files in os.walk(tag_dir):
        for file in files:
            if file.endswith("_label.json"):
                all_json_list.append(os.path.join(root, file))
    # 对文件列表进行排序，确保每次加载顺序一致（支持断点续测）
    all_json_list.sort()
    return all_json_list

def get_all_complate_data(tag_dir, use_cn=False, data_types=None):
    """
    获取所有完整的计数任务数据（支持图片、文本、音频等多模态）
    
    Args:
        tag_dir: 目标目录路径
        use_cn: 是否使用中文问题描述
        data_types: 支持的数据类型列表，如 ["image", "text", "audio"]，默认只加载图片
    
    Returns:
        List[Dict]: 包含所有任务的列表，每个任务包含file_path, question, gt, level, data_type
    """
    if data_types is None:
        data_types = ["image"]
    
    all_json_list = get_all_json(tag_dir)
    all_data = []
    
    for json_path in all_json_list:
        try:
            json_data = json.load(open(json_path, "r", encoding="utf-8"))
            
            # 处理图片数据（原有格式）
            if "type" in json_data and json_data["type"] == "image" and "image" in data_types:
                file_path = json_path.replace(os.path.basename(json_path), json_data["target_file_path"])
                for idx, x in enumerate(json_data["questions"]):
                    question = x["question"]
                    if use_cn and "question_cn" in x: 
                        question = x["question_cn"]
                    all_data.append({
                        "file_path": file_path,
                        "question": question,
                        "gt": x["count"],
                        "level": x["level"],
                        "data_type": "image",
                        "question_id": x.get("question_id"),
                        "annotation_index": idx
                    })
            
            # 处理文本数据（支持多种格式）
            elif "text" in data_types:
                text_content = ""
                document_id = json_path
                questions_list = []
                
                # 格式1：type=text + target_text (new_data/text格式)
                if "type" in json_data and json_data["type"] == "text":
                    text_content = json_data.get("target_text", "")
                    questions_list = json_data.get("questions", [])
                
                # 格式2：document_id + text (其他可能的格式)
                elif "document_id" in json_data and "text" in json_data:
                    text_content = json_data["text"]
                    document_id = json_data["document_id"]
                    questions_list = json_data.get("questions", [])
                
                # 格式3：旧格式的文本数据（可能有其他字段组合）
                elif any(key in json_data for key in ["text_content", "content", "document"]):
                    text_content = json_data.get("text_content") or json_data.get("content") or json_data.get("document", "")
                    questions_list = json_data.get("questions", [])
                    # 如果没有questions但有其他计数信息，创建一个默认问题
                    if not questions_list and any(key in json_data for key in ["count", "total_count"]):
                        default_count = json_data.get("count") or json_data.get("total_count", 0)
                        questions_list = [{
                            "question": "请统计文本中的指定内容数量",
                            "count": default_count,
                            "level": "unknown",
                            "question_id": 1
                        }]
                
                # 处理问题列表
                for idx, question_data in enumerate(questions_list):
                    question = question_data.get("question", "")
                    all_data.append({
                        "file_path": document_id,  # 对于文本，使用JSON文件路径作为标识
                        "question": question,
                        "gt": question_data.get("count", 0),
                        "level": question_data.get("level", "unknown"),
                        "data_type": "text",
                        "text_content": text_content,
                        "question_id": question_data.get("question_id", idx + 1),
                        "annotation_index": idx
                    })
            
            # 处理音频数据（预留接口）
            elif "type" in json_data and json_data["type"] == "audio" and "audio" in data_types:
                file_path = json_path.replace(os.path.basename(json_path), json_data["target_file_path"])
                for idx, x in enumerate(json_data.get("questions", [])):
                    question = x.get("question", "")
                    if use_cn and "question_cn" in x:
                        question = x["question_cn"]
                    all_data.append({
                        "file_path": file_path,
                        "question": question,
                        "gt": x.get("count"),
                        "level": x.get("level", "unknown"),
                        "data_type": "audio",
                        "question_id": x.get("question_id", idx + 1),
                        "annotation_index": idx
                    })
                    
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    return all_data

if __name__ == '__main__':
    # 测试图片数据加载
    print("=== 图片数据测试 ===")
    image_data = get_all_complate_data('../new_data/image', data_types=["image"])
    print(f"图片任务数量: {len(image_data)}")
    if image_data:
        print(f"示例任务: {image_data[0]}")
    
    # 测试多模态数据加载
    print("\n=== 多模态数据测试 ===")
    multi_data = get_all_complate_data('../', data_types=["image", "text", "audio"])
    print(f"多模态任务总数: {len(multi_data)}")
    
    # 按数据类型统计
    type_stats = {}
    for task in multi_data:
        data_type = task.get('data_type', 'unknown')
        type_stats[data_type] = type_stats.get(data_type, 0) + 1
    print(f"数据类型分布: {type_stats}")
