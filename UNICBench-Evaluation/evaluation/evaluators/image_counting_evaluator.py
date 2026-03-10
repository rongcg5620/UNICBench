#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片计数评测器
基于chat_bots.py调用GPT-4o API进行图片计数评测
"""

import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
# 可选依赖导入 - 如果没有安装也能正常运行
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # 创建一个简单的numpy替代
    class SimpleNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5
    np = SimpleNumpy()

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import gc

# 导入chat_bots模块和数据加载器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.chat_bots import ChatBot, ChatBots, readJson, writeJson
from utils.data_loader import get_all_complate_data
import multiprocessing as mp


@dataclass
class ImageCountingTask:
    """图片计数任务数据结构"""
    task_id: str
    category: str           # 任务类别，如 'people_counting'
    image_path: str         # 图片路径
    prompt: str             # 提示词
    ground_truth: Optional[int] = None  # 标准答案
    metadata: Optional[Dict] = None     # 元数据


@dataclass
class CountingResult:
    """计数结果数据结构（扩展版）"""
    task_id: str
    category: str
    image_path: str
    predicted_count: Optional[int]
    ground_truth: Optional[int]
    is_correct: Optional[bool]
    processing_time: float
    error_message: Optional[str] = None
    # 新增指标
    absolute_error: Optional[float] = None
    squared_error: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Optional[str] = None  # 保存原始响应文本用于分析
    response: Optional[Dict] = None  # 保存完整的response对象结构体
    # 关联标注定位信息（用于失败重测精确定位问题）
    annotation_index: Optional[int] = None
    question: Optional[str] = None
    question_id: Optional[int] = None


class NumberExtractor:
    """数字提取器"""
    
    @staticmethod
    def extract_number(text: str) -> Optional[int]:
        """
        从模型的原始文本中提取一个整数。

        优先解析特定格式：
        1. <think>标签后跟着的数字
        2. GLM 风格标记：begin_of_box ... end_of_box
        若未命中，再回退到通用数字提取。

        Args:
            text: API返回的文本
            
        Returns:
            提取到的数字，如果提取失败返回None
        """
        if not text:
            return None

        s = str(text).strip()
        original_text = s  # 保留原始文本用于think标签提取

        # 1) 最优先：提取 think 相关标签后紧跟的数字
        try:
            # 情况1: 匹配完整的 <think>...</think> 后面紧跟的数字
            think_pattern_full = re.compile(r"<think>.*?</think>\s*(\d+)", flags=re.IGNORECASE | re.DOTALL)
            think_match_full = think_pattern_full.search(original_text)
            if think_match_full:
                # 找到了完整 <think>...</think> 标签后面的数字，直接返回
                return int(think_match_full.group(1))
            
            # 情况2: 匹配单独的 </think> 后面紧跟的数字（没有开始标签）
            think_pattern_end = re.compile(r"</think>\s*(\d+)", flags=re.IGNORECASE)
            think_match_end = think_pattern_end.search(original_text)
            if think_match_end:
                # 找到了 </think> 标签后面的数字，直接返回
                return int(think_match_end.group(1))
        except Exception:
            pass

        # 清理显式思维链标签，避免干扰后续解析
        # 注意：只有在上面没有提取到数字时才清理
        try:
            s = re.sub(r"<think>.*?</think>", " ", s, flags=re.IGNORECASE | re.DOTALL)
        except Exception:
            pass

        # 2) 其次：GLM 风格的 box 标记，仅当 box 中存在数字时才认为有效
        try:
            # 抓取所有 begin_of_box ... end_of_box 片段，取"最后一个 box"
            pattern = re.compile(r"begin_of_box\s*(.*?)\s*end_of_box", flags=re.IGNORECASE | re.DOTALL)
            boxes = list(pattern.finditer(s))
            if boxes:
                last_box_content = boxes[-1].group(1)
                nums = re.findall(r"\d+", last_box_content)
                if nums:
                    # 若存在多个数字，取最后一个，通常为最终答案
                    return int(nums[-1])
                # 有 box 但没有数字，视为提取失败
        except Exception:
            pass

        # 3) 纯数字（整串）
        if re.fullmatch(r"\d+", s):
            try:
                return int(s)
            except Exception:
                pass

        # 4) 取消通用回退：若既不是纯数字也没有 box 中的数字，则视为失败（返回 None）
        
        return None


class ImageCountingEvaluator:
    """图片计数评测器"""
    
    def __init__(self, 
                 apis: List[Dict[str, str]], 
                 max_try: int = 10,
                 results_dir: str = 'image_counting_results',
                 test_connectivity: bool = True,
                 bot_indices: Optional[List[int]] = None):
        """
        初始化评测器
        
        Args:
            apis: API配置列表
            max_try: 最大重试次数
            results_dir: 结果保存目录
            test_connectivity: 是否测试模型连通性，默认True
            bot_indices: 要测试的模型索引列表，None表示测试所有模型
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每次运行创建单独的日志文件
        log_file_path = self.results_dir / 'gptCallLog.jsonl.log'
        # 保留原始 apis 以便在子进程内安全构造 ChatBots（Windows 下多进程需要可序列化参数）
        self.apis = apis
        self.chat_bots = ChatBots(apis, max_try=max_try, log_file_path=str(log_file_path))
        self.extractor = NumberExtractor()
        # Remove system_prompt to avoid conflicts with detailed user instructions
        
        # 测试模型连通性
        if test_connectivity:
            print("\n" + "="*80)
            print("图片计数评测器初始化 - 开始测试模型连通性")
            print("="*80)
            connectivity_success, connectivity_msg = self.chat_bots.test_connectivity(bot_indices=bot_indices)
            if not connectivity_success:
                raise ConnectionError(f"模型连通性测试失败，无法开始评测。\n{connectivity_msg}")
            print("连通性测试通过，继续初始化评测器...\n")
        
        # 设置日志，输出到控制台和运行目录下的evaluation.log
        log_file = self.results_dir / 'evaluation.log'
        self.logger = self._setup_logger(log_file)
        
    @staticmethod
    def _model_call_worker(apis, prompt, image_path, system_prompt, per_call_max_try, log_file_path, out_queue):
        """子进程安全地执行一次模型调用，并将结果放入队列。
        返回内容协议：('ok', [content, prompt_tokens, completion_tokens, response_dict]) 或 ('err', str(e))。
        """
        try:
            bots = ChatBots(apis, max_try=per_call_max_try, log_file_path=log_file_path)
            # 该子进程路径目前未传入 question_id，上层如需可扩展；此处写入 None
            res = bots.call(prompt, img=image_path, system_prompt=system_prompt, question_id=None)
            out_queue.put(('ok', res))
        except Exception as e:
            out_queue.put(('err', str(e)))

    def _call_model_with_timeout(self, prompt: str, image_path: str, system_prompt: str, timeout_seconds: int) -> Optional[list]:
        """使用子进程封装一次模型调用，超时后强制终止子进程并返回 None。"""
        try:
            out_q = mp.Queue(maxsize=1)
            # 内层 ChatBot 已改为单次尝试（max_try=1），外层限制为2次；这里保持单次调用
            per_call_max_try = 1
            log_file_path = str(self.results_dir / 'gptCallLog.jsonl.log')
            p = mp.Process(target=ImageCountingEvaluator._model_call_worker,
                           args=(self.apis, prompt, image_path, system_prompt, per_call_max_try, log_file_path, out_q),
                           daemon=True)
            p.start()
            p.join(timeout_seconds)
            if p.is_alive():
                try:
                    p.terminate()
                finally:
                    p.join(5)
                return None
            # 进程已结束，从队列取结果
            if not out_q.empty():
                status, payload = out_q.get_nowait()
                if status == 'ok':
                    return payload
                else:
                    self.logger.error(f"子进程模型调用错误: {payload}")
                    return None
            return None
        except Exception as e:
            self.logger.error(f"模型调用超时封装异常: {e}")
            return None

    def _setup_logger(self, log_file: Optional[Path] = None) -> logging.Logger:
        """设置日志记录器（控制台+文件）"""
        logger_name = f"ImageCountingEvaluator.{self.results_dir.name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # 避免重复输出
        logger.propagate = False
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 控制台输出
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # 文件输出
        if log_file is not None:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                # 文件日志不可用时继续使用控制台日志
                pass

        return logger
    
    def generate_tasks_from_images_dir(self, images_dir: str = "../new_data/image") -> List[ImageCountingTask]:
        """
        从images目录生成计数任务（使用data_loader统一接口）

        Args:
            images_dir: 图片目录路径

        Returns:
            任务列表
        """
        tasks = []
        
        try:
            # 使用data_loader加载图片数据
            all_data = get_all_complate_data(images_dir, data_types=["image"])
            
            if not all_data:
                self.logger.warning(f"未找到图片任务数据: {images_dir}")
                return tasks
            
            # 将data_loader的数据转换为ImageCountingTask
            for i, data in enumerate(all_data):
                # 从文件路径推断类别
                file_path = Path(data['file_path'])
                category = self._infer_category_from_path(file_path)
                
                # 生成task_id
                task_id = f"{category}_{file_path.stem}_{i+1}"
                
                # 从 data 中获取 question_id / annotation_index（若 data_loader 提供）
                qid = data.get('question_id')
                aidx = data.get('annotation_index')

                # 创建任务
                task = ImageCountingTask(
                    task_id=task_id,
                    category=category,
                    image_path=data['file_path'],
                    prompt=data['question'],
                    ground_truth=data['gt'],
                    metadata={
                        'level': data.get('level', 'unknown'),
                        'data_type': data.get('data_type', 'image'),
                        'file_extension': file_path.suffix,
                        'annotation_index': aidx if aidx is not None else i,
                        'question_id': qid
                    }
                )
                tasks.append(task)
            
            self.logger.info(f"生成了 {len(tasks)} 个图片计数任务")
            return tasks
            
        except Exception as e:
            self.logger.error(f"加载图片任务时出错: {str(e)}")
            return tasks
    
    def _infer_category_from_path(self, file_path: Path) -> str:
        """从文件路径推断类别"""
        # 在新的数据组织形式下，类别就是图片文件的父目录名
        return file_path.parent.name if file_path.parent.name else 'unknown'

    def _find_image_files(self, directory: Path) -> List[Path]:
        """递归查找图片文件"""
        image_files = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

        for item in directory.rglob('*'):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                image_files.append(item)

        return image_files
    
    def _load_all_annotations(self, img_file: Path) -> List[Dict[str, Any]]:
        """加载所有标注和问题"""
        # 尝试多种可能的JSON文件位置
        possible_json_paths = [
            # 新格式：*_label.json（和图片在同一目录）
            img_file.parent / f"{img_file.stem}_label.json",
            # 标准位置：和images同级的jsons目录
            img_file.parent.parent / "jsons" / f"{img_file.stem}.json",
            # 直接同名文件
            img_file.with_suffix('.json'),
            # 上级目录的jsons
            img_file.parent / "jsons" / f"{img_file.stem}.json",
            # 上上级目录的jsons
            img_file.parent.parent.parent / "jsons" / f"{img_file.stem}.json",
            # 新格式在jsons目录中
            img_file.parent.parent / "jsons" / f"{img_file.stem}_label.json",
            img_file.parent / "jsons" / f"{img_file.stem}_label.json"
        ]
        
        json_file = None
        for path in possible_json_paths:
            if path.exists():
                json_file = path
                break
        
        if json_file is None:
            self.logger.debug(f"未找到标注文件，尝试的路径: {[str(p) for p in possible_json_paths]}")
            return []

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            annotations = []

            # 处理新格式标注文件：questions数组中的count和question字段
            if 'questions' in data and isinstance(data['questions'], list):
                for i, question_data in enumerate(data['questions']):
                    count = None
                    question = None
                    qid = None
                    
                    if 'count' in question_data:
                        count = int(question_data['count'])
                    elif 'instances' in question_data and isinstance(question_data['instances'], list):
                        count = len(question_data['instances'])

                    if 'question' in question_data:
                        question = question_data['question']
                    if 'question_id' in question_data:
                        try:
                            qid = int(question_data['question_id'])
                        except Exception:
                            qid = None
                    
                    if count is not None or question is not None:
                        annotations.append({
                            'count': count,
                            'question': question,
                            'annotation_index': i,
                            'question_id': qid
                        })
            
            # 处理旧格式标注文件：annotations数组中的count和question字段
            elif 'annotations' in data and isinstance(data['annotations'], list):
                for i, annotation in enumerate(data['annotations']):
                    count = None
                    question = None
                    
                    if 'count' in annotation:
                        count = int(annotation['count'])
                    elif 'points' in annotation:
                        count = len(annotation['points'])

                    if 'question' in annotation:
                        question = annotation['question']
                    
                    if count is not None or question is not None:
                        annotations.append({
                            'count': count,
                            'question': question,
                            'annotation_index': i,
                            'question_id': None
                        })

            # 如果没有annotations或questions数组，尝试其他可能的结构
            else:
                # 尝试直接从根级别获取count和question
                count = None
                question = None
                
                if 'count' in data:
                    count = int(data['count'])
                elif 'total_count' in data:
                    count = int(data['total_count'])
                elif 'num_objects' in data:
                    count = int(data['num_objects'])
                elif 'objects' in data and isinstance(data['objects'], list):
                    count = len(data['objects'])
                    
                if 'question' in data:
                    question = data['question']
                elif 'task' in data:
                    question = data['task']
                    
                if count is not None:
                    annotations.append({
                        'count': count,
                        'question': question,
                        'annotation_index': 0,
                        'question_id': None
                    })

            if annotations:
                self.logger.debug(f"成功加载标注文件: {json_file}, 找到 {len(annotations)} 个标注")
            else:
                self.logger.warning(f"标注文件 {json_file} 中未找到有效的count数据")
                
            return annotations

        except Exception as e:
            self.logger.error(f"读取标注文件 {json_file} 出错: {e}")
            return []

    def _load_ground_truth_and_question(self, img_file: Path) -> tuple[Optional[int], Optional[str]]:
        """加载ground truth标注和问题（保留向后兼容性）"""
        annotations = self._load_all_annotations(img_file)
        if annotations:
            first_annotation = annotations[0]
            return first_annotation['count'], first_annotation['question']
        return None, None
    
    def _validate_zero_count_result(self, task: ImageCountingTask, raw_response: str) -> bool:
        """
        验证零计数结果是否合理
        
        Args:
            task: 计数任务
            raw_response: API原始响应
            
        Returns:
            零计数是否合理
        """
        # 纯数字"0"的回复直接视为合理
        try:
            if raw_response is not None and raw_response.strip() == "0":
                return True
        except Exception:
            pass
        
        # 对于细胞计数任务，零计数可能合理
        if task.category.lower() == 'cell':
            # 检查响应中是否包含相关关键词
            response_lower = raw_response.lower()
            zero_indicators = ['no', 'none', 'zero', '0', 'empty', 'clear', 'background']
            has_zero_indicator = any(indicator in response_lower for indicator in zero_indicators)
            
            if has_zero_indicator:
                self.logger.info(f"任务 {task.task_id}: 零计数结果合理，响应包含零计数指示词")
                return True
            else:
                self.logger.warning(f"任务 {task.task_id}: 零计数结果可能不合理，响应: {raw_response}")
                return False
        
        return True

    def evaluate_single_task(self, task: ImageCountingTask) -> CountingResult:
        """
        评测单个任务

        Args:
            task: 计数任务

        Returns:
            计数结果
        """
        start_time = time.time()

        # 记录内存使用情况（如果psutil可用）
        if HAS_PSUTIL:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        else:
            memory_before = 0

        try:
            self.logger.info(f"评测任务: {task.task_id}")

            # 提取失败/拒答的最大重试次数：5（含首次）
            max_extract_try = 5
            extracted_count = None
            raw_response = None
            response_dict = None  # 保存完整的response对象
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            error_message = None

            # 统一的系统提示词
            system_prompt = (
                "You are a counting assistant. You MUST respond with ONLY a number. "
                "Never refuse to answer. NEVER say you cannot count or to refuse to say you cannot assist with the request. "
                "Always give your best numerical estimate. Respond with just the number, nothing else."
            )
            
            # 外层：仅负责"提取失败/格式错误"的重试
            for extract_try in range(max_extract_try):
                # 简单使用原始问题，不添加额外指令
                user_txt = task.prompt

                # 直接调用API
                qid = (task.metadata or {}).get('question_id')
                result = self.chat_bots.call(user_txt, img=task.image_path, system_prompt=system_prompt, question_id=qid)
                processing_time = time.time() - start_time

                # 计算内存使用
                if HAS_PSUTIL:
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage = memory_after - memory_before
                else:
                    memory_usage = 0

                if result is None:
                    # API调用失败，直接报错
                    error_message = "API调用失败"
                    break

                raw_response, prompt_tokens, completion_tokens, response_dict = result
                
                # 检查是否是模型限制错误
                if isinstance(raw_response, str) and raw_response.startswith("[MODEL_LIMIT_ERROR]"):
                    error_message = raw_response
                    # 这是模型能力限制，不是评测程序的错误
                    self.logger.warning(f"模型context限制: {error_message}")
                    break
                total_tokens = prompt_tokens + completion_tokens

                # 提取数字
                extracted_count = self.extractor.extract_number(raw_response)
                if extracted_count is not None:
                    # 成功提取数字，结束重试
                    break
                
                # 提取失败，判断是否应该重试
                
                # 1. 检查是否是错误信息（这些错误在 chat_bots 层已经重试过了，不应该再重试）
                is_error_response = (
                    isinstance(raw_response, str) and (
                        raw_response.startswith("[RATE_LIMIT_ERROR]") or
                        raw_response.startswith("[SERVER_ERROR]") or
                        raw_response.startswith("[API_ERROR]") or
                        raw_response.startswith("[TIMEOUT_ERROR]") or
                        raw_response.startswith("[CONNECTION_ERROR]")
                    )
                )
                
                if is_error_response:
                    # chat_bots 层已经重试过的错误，直接结束
                    self.logger.warning(f"收到错误响应，不再重试: {raw_response[:100]}...")
                    break
                
                # 2. 模型有回复但提取不到数字，继续重试（最多5次）
                self.logger.warning(f"未能从回复中提取数字（第{extract_try+1}/{max_extract_try}次），回复内容: {raw_response[:200] if raw_response else 'None'}...")
                
                # 如果已经是最后一次尝试，记录详细信息
                if extract_try == max_extract_try - 1:
                    self.logger.error(f"达到最大重试次数({max_extract_try})，仍无法提取数字。最后回复: {raw_response}")

            # 计算各种指标
            is_correct = None
            absolute_error = None
            squared_error = None

            # 记录无法提取数字的样本到 warning（包含无响应/异常场景）
            if extracted_count is None:
                warning_path = self.results_dir / 'warning_images.txt'
                with open(warning_path, 'a', encoding='utf-8') as wf:
                    ann_idx = (task.metadata or {}).get('annotation_index')
                    qid = (task.metadata or {}).get('question_id')
                    qtext = task.prompt
                    safe_qtext = (qtext or '').replace('\t', ' ').replace('\n', ' ')
                    safe_resp = (raw_response or '').replace('\t', ' ').replace('\n', ' ')
                    # 如果是模型限制错误，添加特殊标记
                    if error_message and "[MODEL_LIMIT_ERROR]" in str(error_message):
                        safe_resp = f"[模型Context超限] {safe_resp}"
                    wf.write(f"{task.image_path}\t{ann_idx}\t{qid}\t{safe_qtext}\t{safe_resp}\n")

            # 验证零计数结果的合理性
            if extracted_count == 0 and raw_response is not None:
                is_zero_reasonable = self._validate_zero_count_result(task, raw_response)
                if not is_zero_reasonable:
                    self.logger.warning(f"任务 {task.task_id}: 零计数结果可能有问题，建议人工检查")

            if task.ground_truth is not None and extracted_count is not None:
                is_correct = extracted_count == task.ground_truth
                absolute_error = abs(extracted_count - task.ground_truth)
                squared_error = (extracted_count - task.ground_truth) ** 2
                

            return CountingResult(
                task_id=task.task_id,
                category=task.category,
                image_path=task.image_path,
                predicted_count=extracted_count,
                ground_truth=task.ground_truth,
                is_correct=is_correct,
                processing_time=processing_time,
                absolute_error=absolute_error,
                squared_error=squared_error,
                memory_usage_mb=memory_usage,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=raw_response,
                response=response_dict,  # 保存完整的response对象
                error_message=error_message,
                annotation_index=(task.metadata or {}).get('annotation_index'),
                question=task.prompt,
                question_id=(task.metadata or {}).get('question_id')
            )

        except Exception as e:
            processing_time = time.time() - start_time
            if HAS_PSUTIL:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0

            self.logger.error(f"任务 {task.task_id} 评测失败: {str(e)}")

            # 异常也写入 warning，便于后续自动复测
            try:
                warning_path = self.results_dir / 'warning_images.txt'
                with open(warning_path, 'a', encoding='utf-8') as wf:
                    ann_idx = (task.metadata or {}).get('annotation_index')
                    qid = (task.metadata or {}).get('question_id')
                    qtext = task.prompt
                    safe_qtext = (qtext or '').replace('\t', ' ').replace('\n', ' ')
                    safe_err = str(e).replace('\t', ' ').replace('\n', ' ')
                    wf.write(f"{task.image_path}\t{ann_idx}\t{qid}\t{safe_qtext}\t{safe_err}\n")
            except Exception:
                pass

            return CountingResult(
                task_id=task.task_id,
                category=task.category,
                image_path=task.image_path,
                predicted_count=None,
                ground_truth=task.ground_truth,
                is_correct=None,
                processing_time=processing_time,
                memory_usage_mb=memory_usage,
                error_message=str(e),
                annotation_index=(task.metadata or {}).get('annotation_index'),
                question=task.prompt,
                question_id=(task.metadata or {}).get('question_id')
            )

    def evaluate_batch(self, tasks: List[ImageCountingTask], save_intermediate: bool = False) -> List[CountingResult]:
        """
        批量评测图片计数任务

        Args:
            tasks: 任务列表
            save_intermediate: 是否保存中间结果（文件名基于图片编号）

        Returns:
            结果列表
        """
        results = []
        total_tasks = len(tasks)
        
        self.logger.info(f"开始批量评测 {total_tasks} 个任务...")
        
        last_saved_index = 0  # 记录上次中间结果保存的位置（增量起点）
        for i, task in enumerate(tasks):
            try:
                self.logger.info(f"评测任务 [{i+1}/{total_tasks}]: {task.task_id}")
                result = self.evaluate_single_task(task)
                results.append(result)
                
                # 定期保存中间结果（仅保存自上次保存以来的增量），默认关闭
                # 文件名会自动基于最后一个图片的编号
                if save_intermediate and ((i + 1) % 10 == 0 or (i + 1) == total_tasks):
                    incremental_results = results[last_saved_index:]
                    if incremental_results:
                        # filename 参数仅作为回退方案，实际会从图片路径提取编号
                        self._save_intermediate_results(incremental_results, str(i + 1))
                        last_saved_index = len(results)
                    
            except Exception as e:
                self.logger.error(f"评测任务 {task.task_id} 失败: {str(e)}")
                # 记录错误结果
                error_result = CountingResult(
                    task_id=task.task_id,
                    category=task.category,
                    image_path=task.image_path,
                    predicted_count=None,
                    ground_truth=task.ground_truth,
                    is_correct=None,
                    processing_time=0.0,
                    error_message=str(e)
                )
                results.append(error_result)
        
        self.logger.info(f"批量评测完成，共 {len(results)} 个结果")
        return results

    def _extract_image_number(self, image_path: str) -> Optional[int]:
        """从图片路径中提取编号（如 0001.jpg -> 1, 0060.jpg -> 60）"""
        try:
            from pathlib import Path
            # 获取文件名（不含扩展名）
            filename = Path(image_path).stem
            # 尝试提取数字
            import re
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def _save_intermediate_results(self, results: List[CountingResult], filename: str):
        """
        保存中间结果增量（仅写入本批新增的结果）。
        在当前运行目录下为每个类别创建子文件夹，并将本批增量写入
        "{category}_batch_progress_{category}_{image_number}.json"。
        文件名中的数字基于最后一个图片的编号。
        """
        # 按类别分组当前这批增量结果
        results_by_category = {}
        for result in results:
            category = result.category
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        # 写出增量文件
        results_root = self.results_dir
        results_root.mkdir(exist_ok=True, parents=True)
        for category, category_results in results_by_category.items():
            category_dir = results_root / f"{category}_results"
            category_dir.mkdir(exist_ok=True)
            
            # 从最后一个结果中提取图片编号作为文件名
            last_result = category_results[-1]
            image_number = self._extract_image_number(last_result.image_path)
            
            # 如果无法提取编号，回退到使用传入的 filename
            if image_number is not None:
                category_filename = f"{category}_batch_progress_{category}_{image_number}.json"
            else:
                category_filename = f"{category}_batch_progress_{filename}.json"
            
            save_path = category_dir / category_filename
            results_data = []
            for result in category_results:
                results_data.append({
                    "task_id": result.task_id,
                    "image_path": result.image_path,
                    "predicted_count": result.predicted_count,
                    "ground_truth": result.ground_truth,
                    "is_correct": result.is_correct,
                    "processing_time": result.processing_time,
                    "absolute_error": result.absolute_error,
                    "squared_error": result.squared_error,
                    "memory_usage_mb": result.memory_usage_mb,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "raw_response": result.raw_response,
                    "response": result.response,  # 完整的response对象
                    # 不输出 error_message 到中间文件
                    "annotation_index": result.annotation_index,
                    "question": result.question,
                    "question_id": result.question_id
                })
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)

    def evaluate_by_category(self, tasks: List[ImageCountingTask], save_after_each: bool = True, save_intermediate: bool = False) -> List[CountingResult]:
        """
        按类别分组评测；每个类别完成后可立即保存该类别的完整结果文件。

        Args:
            tasks: 待评测任务
            save_after_each: 是否在每个类别完成后保存该类别的完整结果
            save_intermediate: 是否保存中间结果（文件名基于图片编号）

        Returns:
            全部结果列表
        """
        if not tasks:
            return []

        # 分组
        tasks_by_category: Dict[str, List[ImageCountingTask]] = {}
        for t in tasks:
            tasks_by_category.setdefault(t.category, []).append(t)

        # 初始化进度跟踪变量
        total_tasks = len(tasks)
        completed_tasks = 0
        start_time = time.time()

        all_results: List[CountingResult] = []
        # 逐类别评测
        for category, cat_tasks in tasks_by_category.items():
            self.logger.info(f"按类别评测开始: {category}（{len(cat_tasks)} 个任务）")
            category_results: List[CountingResult] = []

            last_saved_index = 0
            for i, task in enumerate(cat_tasks):
                try:
                    # 计算进度信息
                    current_percentage = completed_tasks / total_tasks * 100
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks if completed_tasks > 0 else 0
                    remaining_tasks = total_tasks - completed_tasks
                    eta_seconds = avg_time_per_task * remaining_tasks
                    
                    if eta_seconds > 0:
                        eta_str = f", 预计剩余: {int(eta_seconds//60)}分{int(eta_seconds%60)}秒"
                    else:
                        eta_str = ""
                    
                    self.logger.info(f"[{category}] 评测任务 [{i+1}/{len(cat_tasks)}]: {task.task_id} | 总进度: {current_percentage:.1f}% ({completed_tasks}/{total_tasks}){eta_str}")
                    
                    result = self.evaluate_single_task(task)
                    category_results.append(result)
                    completed_tasks += 1

                    # 类别内按固定频率保存增量中间结果（默认关闭）
                    # 文件名会自动基于最后一个图片的编号
                    if save_intermediate and ((i + 1) % 10 == 0 or (i + 1) == len(cat_tasks)):
                        incremental = category_results[last_saved_index:]
                        if incremental:
                            # filename 参数仅作为回退方案，实际会从图片路径提取编号
                            self._save_intermediate_results(incremental, f"{category}_{i+1}")
                            last_saved_index = len(category_results)

                except Exception as e:
                    self.logger.error(f"[{category}] 任务 {task.task_id} 失败: {str(e)}")
                    error_result = CountingResult(
                        task_id=task.task_id,
                        category=task.category,
                        image_path=task.image_path,
                        predicted_count=None,
                        ground_truth=task.ground_truth,
                        is_correct=None,
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    category_results.append(error_result)
                    completed_tasks += 1  # 也要增加已完成任务计数

            # 类别完成后显示统计并保存完整类别结果
            if category_results:
                # 显示类别完成进度
                category_percentage = completed_tasks / total_tasks * 100
                elapsed_time = time.time() - start_time
                self.logger.info(f"类别 {category} 完成 | 总进度: {category_percentage:.1f}% ({completed_tasks}/{total_tasks}), 已用时: {int(elapsed_time//60)}分{int(elapsed_time%60)}秒")
                
                # 保存结果
                if save_after_each:
                    self.save_results(category_results)
                    self.logger.info(f"按类别保存完成: {category}")

            all_results.extend(category_results)

        # 显示最终完成统计
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"图像评测全部完成!")
        self.logger.info(f"总任务数: {total_tasks}")
        self.logger.info(f"总用时: {int(total_time//60)}分{int(total_time%60)}秒")
        if total_tasks > 0:
            self.logger.info(f"平均每个任务: {total_time/total_tasks:.2f}秒")
        self.logger.info("=" * 60)
        
        return all_results

    def generate_report(self, results: List[CountingResult]) -> Dict[str, Any]:
        """
        生成评测报告

        Args:
            results: 结果列表

        Returns:
            评测报告
        """
        if not results:
            return {"error": "没有结果数据"}

        total_tasks = len(results)
        successful_extractions = sum(1 for r in results if r.predicted_count is not None)
        total_tokens_used = sum((r.total_tokens or 0) for r in results)

        # 计算准确率（只考虑有ground truth的任务）
        tasks_with_gt = [r for r in results if r.ground_truth is not None]
        correct_predictions = sum(1 for r in tasks_with_gt if r.is_correct is True)
        accuracy = correct_predictions / len(tasks_with_gt) if tasks_with_gt else 0

        # 计算平均处理时间
        avg_processing_time = sum(r.processing_time for r in results) / total_tasks

        # 按类别分组统计
        category_stats = {}
        for result in results:
            category = result.category
            if category not in category_stats:
                category_stats[category] = {
                    'total_questions': 0,
                    'total_images': set(),
                    'successful': 0,
                    'correct': 0,
                    'with_gt': 0
                }

            category_stats[category]['total_questions'] += 1
            category_stats[category]['total_images'].add(result.image_path)
            if result.predicted_count is not None:
                category_stats[category]['successful'] += 1
            if result.ground_truth is not None:
                category_stats[category]['with_gt'] += 1
                if result.is_correct is True:
                    category_stats[category]['correct'] += 1

        # 计算各类别准确率和转换 total_images 为数量
        for category in category_stats:
            stats = category_stats[category]
            stats['total_images'] = len(stats['total_images'])  # 转换为数量
            stats['total'] = stats['total_questions']  # 保持向后兼容
            stats['accuracy'] = stats['correct'] / stats['with_gt'] if stats['with_gt'] > 0 else 0
            stats['extraction_rate'] = stats['successful'] / stats['total_questions'] if stats['total_questions'] > 0 else 0

        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_extractions": successful_extractions,
                "extraction_rate": successful_extractions / total_tasks,
                "tasks_with_ground_truth": len(tasks_with_gt),
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
                "average_processing_time": avg_processing_time,
                "total_tokens": total_tokens_used
            },
            "by_category": category_stats,
            "timestamp": datetime.now().isoformat()
        }

        return report

    def save_results(self, results: List[CountingResult], filename: str = None):
        """
        保存评测结果，按类别创建文件夹

        Args:
            results: 结果列表
            filename: 文件名，如果为None则自动生成
        """
        # 按类别分组结果
        results_by_category = {}
        for result in results:
            category = result.category
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []

        # 为每个类别创建文件夹并保存结果
        for category, category_results in results_by_category.items():
            # 创建类别文件夹
            category_dir = self.results_dir / f"{category}_results"
            category_dir.mkdir(exist_ok=True)

            # 生成文件名
            if filename is None:
                category_filename = f"{category}_results_{timestamp}.json"
            else:
                category_filename = f"{category}_{filename}"

            save_path = category_dir / category_filename

            # 保存完整的结果数据
            results_data = []
            for result in category_results:
                results_data.append({
                    "task_id": result.task_id,
                    "image_path": result.image_path,
                    "predicted_count": result.predicted_count,
                    "ground_truth": result.ground_truth,
                    "is_correct": result.is_correct,
                    "processing_time": result.processing_time,
                    "absolute_error": result.absolute_error,
                    "squared_error": result.squared_error,
                    "memory_usage_mb": result.memory_usage_mb,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "raw_response": result.raw_response,
                    "response": result.response,  # 完整的response对象
                    # 移除 error_message 按需不输出
                    # "error_message": result.error_message,
                    "annotation_index": result.annotation_index,
                    "question": result.question,
                    "question_id": result.question_id
                })

            # 保存结果文件
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"{category} 类别结果已保存到: {save_path}")
            saved_paths.append(save_path)

        return saved_paths
