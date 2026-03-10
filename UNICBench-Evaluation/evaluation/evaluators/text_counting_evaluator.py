#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本计数评测器
基于chat_bots.py调用GPT-4o API进行文本计数评测
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

# 可选依赖导入
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
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

# 导入chat_bots模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.chat_bots import ChatBot, ChatBots, readJson, writeJson
from utils.data_loader import get_all_complate_data

# 可选导入本地API配置
try:
    from gpt4o_config import GPT4O_API_CONFIG
    HAS_LOCAL_CONFIG = True
except ImportError:
    HAS_LOCAL_CONFIG = False
    GPT4O_API_CONFIG = None


# 统一的系统提示词
DEFAULT_SYSTEM_PROMPT = (
    "You are a counting assistant. You MUST respond with ONLY a number. "
    "Never refuse to answer. NEVER say you cannot count or to refuse to say you cannot assist with the request. "
    "Always give your best numerical estimate. Respond with just the number, nothing else."
)


class TextLanguageDetector:
    """文本语言检测器"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        检测文本主要语言
        
        Args:
            text: 要检测的文本
            
        Returns:
            'zh' 表示中文，'en' 表示英文
        """
        if not text:
            return 'en'  # 默认英文
        
        # 统计中文字符比例
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if char.strip():  # 忽略空白字符
                total_chars += 1
                # 检查是否为中文字符（包括中文标点）
                if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                    chinese_chars += 1
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        
        # 如果中文字符超过30%，认为是中文文本
        return 'zh' if chinese_ratio > 0.3 else 'en'
    
    @staticmethod
    def is_chinese_text(text: str) -> bool:
        """判断是否为中文文本"""
        return TextLanguageDetector.detect_language(text) == 'zh'



class TextNumberExtractor:
    """文本数字提取器"""
    
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
            # 支持两种格式：begin_of_box 或 <|begin_of_box|>
            pattern = re.compile(r"(?:<\|)?begin_of_box(?:\|>)?\s*(.*?)\s*(?:<\|)?end_of_box(?:\|>)?", flags=re.IGNORECASE | re.DOTALL)
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

        # 如果没有匹配到任何规范格式，返回None
        # 不再进行通用数字提取，避免从错误信息中提取数字
        return None

@dataclass
class TextCountingTask:
    """文本计数任务数据结构"""
    document_id: str
    question: str
    text_content: str
    ground_truth: int
    level: str
    question_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TextEvaluationResult:
    """文本评测结果数据结构（扩展版）"""
    task: TextCountingTask
    predicted_count: int
    raw_response: str
    is_correct: bool
    absolute_error: float
    relative_error: float
    processing_time: float
    # 新增指标
    squared_error: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error_message: Optional[str] = None
    response: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['task'] = self.task.to_dict()
        return result

class TextCountingEvaluator:
    """文本计数评测器"""
    
    def __init__(self, 
                 apis: List[Dict[str, str]], 
                 max_try: int = 6,
                 save_dir: str = "results",
                 batch_mode: bool = True,
                 test_connectivity: bool = True,
                 bot_indices: Optional[List[int]] = None):
        """
        初始化评测器
        
        Args:
            apis: API配置列表
            max_try: 最大重试次数
            save_dir: 结果保存目录
            batch_mode: 是否启用批量问题处理模式
            test_connectivity: 是否测试模型连通性，默认True
            bot_indices: 要测试的模型索引列表，None表示测试所有模型
                        例如: [0] 只测试第一个模型
        """
        self.apis = apis
        self.max_try = max_try
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.batch_mode = batch_mode
        
        # 初始化ChatBots
        log_file_path = self.save_dir / 'gptCallLog.jsonl.log'
        self.chat_bots = ChatBots(apis, max_try=max_try, log_file_path=str(log_file_path))
        self.extractor = TextNumberExtractor()
        
        # 测试模型连通性
        if test_connectivity:
            print("\n" + "="*80)
            print("文本计数评测器初始化 - 开始测试模型连通性")
            print("="*80)
            connectivity_success, connectivity_msg = self.chat_bots.test_connectivity(bot_indices=bot_indices)
            if not connectivity_success:
                raise ConnectionError(f"模型连通性测试失败，无法开始评测。\n{connectivity_msg}")
            print("连通性测试通过，继续初始化评测器...\n")
        
        # 设置日志
        self._setup_logging()
        
        # 加载已存在的warning记录，避免重复写入
        self.existing_warnings = set()
        self._load_existing_warnings()
        
        # 评测统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'start_time': None,
            'end_time': None,
            'peak_memory_mb': 0,
            'average_memory_mb': 0,
            'total_tokens': 0,
            'average_processing_time': 0,
            'batch_calls': 0,
            'single_calls': 0
        }
        self._memory_samples = []
    
    def _setup_logging(self):
        """设置日志配置"""
        log_file = self.save_dir / f"text_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 创建logger
        logger_name = f"TextCountingEvaluator.{self.save_dir.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # 避免重复输出
        self.logger.propagate = False
        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 控制台输出
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # 文件输出
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            # 文件日志不可用时继续使用控制台日志
            pass
    
    def _load_existing_warnings(self):
        """加载已存在的warning记录"""
        warning_path = self.save_dir / 'warning_texts.txt'
        if warning_path.exists():
            try:
                with open(warning_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                # 使用(document_id, question_id)作为唯一标识
                                doc_id = parts[0]
                                q_id = parts[1]
                                self.existing_warnings.add((doc_id, q_id))
                self.logger.info(f"加载了 {len(self.existing_warnings)} 条已存在的warning记录")
            except Exception as e:
                self.logger.warning(f"加载warning记录失败: {e}")
    
    def load_text_tasks(self, text_dir: str, categories: List[str] = None) -> List[TextCountingTask]:
        """
        加载文本计数任务
        
        Args:
            text_dir: 文本数据目录
            categories: 指定的文本类别，如果为None则加载所有类别
            
        Returns:
            List[TextCountingTask]: 文本计数任务列表
        """
        self.logger.info(f"Loading text tasks from: {text_dir}")
        
        # 使用data_loader加载文本数据
        all_data = get_all_complate_data(text_dir, data_types=["text"])
        
        tasks = []
        for data in all_data:
            # 如果指定了类别，进行过滤
            if categories:
                # 从文件路径中提取类别信息
                category_matched = False
                for category in categories:
                    if category in data['file_path']:
                        category_matched = True
                        break
                if not category_matched:
                    continue
            
            task = TextCountingTask(
                document_id=data['file_path'],
                question=data['question'],
                text_content=data['text_content'],
                ground_truth=data['gt'],
                level=data['level'],
                question_id=data.get('question_id', 0)
            )
            tasks.append(task)
        
        self.logger.info(f"Loaded {len(tasks)} text counting tasks")
        return tasks
    
    def _create_prompt_from_question(self, question: str, text_content: str, category: Optional[str] = None) -> str:
        """从问题和文本内容创建prompt（自动检测语言）"""
        # 自动检测语言并选择合适的prompt
        text_is_chinese = TextLanguageDetector.is_chinese_text(text_content)
        question_is_chinese = TextLanguageDetector.is_chinese_text(question)
        
        # 决定使用的语言
        use_chinese = question_is_chinese or (text_is_chinese and not question_is_chinese)
        
        # 不对文本长度做任何限制，让模型处理完整文本
        # 如果文本超过模型的context window，那是模型本身的限制，不是评测程序的问题
        text_display = (text_content)
        
        # 构建最终prompt（不再包含详细的格式要求，由系统提示词统一处理）
        if use_chinese:
            return f"""文本内容：
{text_display}

问题：{question}"""
        else:
            return f"""Text Content:
{text_display}

Question: {question}"""
    
    def evaluate_single_task(self, task: TextCountingTask) -> TextEvaluationResult:
        """
        评测单个文本计数任务（与图像评测器保持一致的重试逻辑）
        
        Args:
            task: 文本计数任务
            
        Returns:
            TextEvaluationResult: 评测结果
        """
        start_time = time.time()
        
        # 记录内存使用情况（如果psutil可用）
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                memory_before = 0
        else:
            memory_before = 0

        try:
            self.logger.info(f"评测文本任务: {task.question_id}")

            # 提取失败/拒答的最大重试次数：5（含首次）
            max_extract_try = 5
            extracted_count = None
            raw_response = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            error_message = None

            # 获取任务类别
            category = self._get_task_category(task)

            # 检测语言用于重试prompt
            text_is_chinese = TextLanguageDetector.is_chinese_text(task.text_content[:500])  # 只检测前500字符提高性能
            question_is_chinese = TextLanguageDetector.is_chinese_text(task.question)
            use_chinese = question_is_chinese or (text_is_chinese and not question_is_chinese)
            
            # 记录语言检测结果（debug级别）
            self.logger.debug(f"语言检测 - 文本: {'中文' if text_is_chinese else '英文'}, 问题: {'中文' if question_is_chinese else '英文'}, 使用: {'中文' if use_chinese else '英文'}prompt")
            
            # 外层：仅负责"提取失败/格式错误"的重试
            for extract_try in range(max_extract_try):
                # 简单构造用户提示，不再强化
                user_txt = self._create_prompt_from_question(task.question, task.text_content, category)
                
                # 使用统一的系统提示词
                system_prompt = DEFAULT_SYSTEM_PROMPT

                # 调用API
                result = self.chat_bots.call(user_txt, system_prompt=system_prompt, question_id=task.question_id)
                processing_time = time.time() - start_time

                # 计算内存使用
                if HAS_PSUTIL:
                    try:
                        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        memory_usage = memory_after - memory_before
                    except Exception:
                        memory_usage = 0
                else:
                    memory_usage = 0

                if result is None:
                    # API调用失败，直接报错
                    error_message = "API调用失败"
                    break

                raw_response, prompt_tokens, completion_tokens, response_dict = result
                
                # 检查是否是模型限制错误或速率限制错误
                if isinstance(raw_response, str) and (raw_response.startswith("[MODEL_LIMIT_ERROR]") or 
                                                       raw_response.startswith("[RATE_LIMIT_ERROR]")):
                    error_message = raw_response
                    # 这是模型限制或速率限制，不是评测程序的错误
                    self.logger.warning(f"模型context限制: {error_message}")
                    break
                total_tokens = prompt_tokens + completion_tokens

                # 提取数字
                extracted_count = self.extractor.extract_number(raw_response)
                if extracted_count is not None:
                    break
                else:
                    self.logger.warning(f"未能从回复中提取数字（第{extract_try+1}/{max_extract_try}次），回复内容: {raw_response}")

            # 记录无法提取数字的样本到 warning（包含无响应/异常场景）
            if extracted_count is None:
                # 检查是否已经记录过这个失败任务
                warning_key = (task.document_id, str(task.question_id))
                if warning_key not in self.existing_warnings:
                    try:
                        warning_path = self.save_dir / 'warning_texts.txt'
                        safe_q = (task.question or '').replace('\t', ' ').replace('\n', ' ')
                        safe_resp = (raw_response or '').replace('\t', ' ').replace('\n', ' ')
                        # 如果是模型限制错误或速率限制错误，添加特殊标记
                        if error_message and "[MODEL_LIMIT_ERROR]" in str(error_message):
                            safe_resp = f"[模型Context超限] {safe_resp}"
                        elif error_message and "[RATE_LIMIT_ERROR]" in str(error_message):
                            # 提取平台信息（Azure或其他）
                            if "TPM limit exceeded on Azure" in str(error_message):
                                safe_resp = f"[TPM超Azure上限] {safe_resp}"
                            elif "TPM limit exceeded on API" in str(error_message):
                                safe_resp = f"[TPM超API上限] {safe_resp}"
                            elif "TPM limit exceeded" in str(error_message):
                                safe_resp = f"[TPM超限] {safe_resp}"
                            else:
                                safe_resp = f"[Token速率超限] {safe_resp}"
                        with open(warning_path, 'a', encoding='utf-8') as wf:
                            wf.write(f"{task.document_id}\t{task.question_id}\t{safe_q}\t{safe_resp}\n")
                        # 添加到已记录集合中
                        self.existing_warnings.add(warning_key)
                    except Exception:
                        pass

            # 计算各种指标
            is_correct = None
            absolute_error = None
            squared_error = None
            relative_error = None

            if task.ground_truth is not None and extracted_count is not None:
                is_correct = extracted_count == task.ground_truth
                absolute_error = abs(extracted_count - task.ground_truth)
                squared_error = (extracted_count - task.ground_truth) ** 2
                relative_error = absolute_error / max(task.ground_truth, 1)
            elif extracted_count is not None:
                # 没有ground truth但提取成功
                is_correct = None
                absolute_error = None
                squared_error = None
                relative_error = None
            else:
                # 提取失败
                is_correct = False
                absolute_error = float('inf')
                squared_error = float('inf')
                relative_error = float('inf')
                extracted_count = -1

            return TextEvaluationResult(
                task=task,
                predicted_count=extracted_count,
                raw_response=raw_response or "",
                is_correct=is_correct if is_correct is not None else False,
                absolute_error=absolute_error if absolute_error is not None else float('inf'),
                relative_error=relative_error if relative_error is not None else float('inf'),
                processing_time=processing_time,
                squared_error=squared_error if squared_error is not None else float('inf'),
                memory_usage_mb=memory_usage,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response=response_dict
            )

        except Exception as e:
            processing_time = time.time() - start_time
            if HAS_PSUTIL:
                try:
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    memory_usage = memory_after - memory_before
                except Exception:
                    memory_usage = 0
            else:
                memory_usage = 0

            self.logger.error(f"任务 {task.question_id} 评测失败: {str(e)}")

            # 异常也写入 warning，便于后续自动复测
            warning_key = (task.document_id, str(task.question_id))
            if warning_key not in self.existing_warnings:
                try:
                    warning_path = self.save_dir / 'warning_texts.txt'
                    safe_q = (task.question or '').replace('\t', ' ').replace('\n', ' ')
                    safe_err = str(e).replace('\t', ' ').replace('\n', ' ')
                    with open(warning_path, 'a', encoding='utf-8') as wf:
                        wf.write(f"{task.document_id}\t{task.question_id}\t{safe_q}\t{safe_err}\n")
                    # 添加到已记录集合中
                    self.existing_warnings.add(warning_key)
                except Exception:
                    pass

            return TextEvaluationResult(
                task=task,
                predicted_count=-1,
                raw_response=f"Error: {str(e)}",
                is_correct=False,
                absolute_error=float('inf'),
                relative_error=float('inf'),
                processing_time=processing_time,
                squared_error=float('inf'),
                memory_usage_mb=memory_usage,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error_message=str(e),
                response=None
            )
    
    def _get_task_category(self, task: TextCountingTask) -> str:
        """从任务推断类别"""
        try:
            return Path(task.document_id).parent.name if task.document_id else 'unknown'
        except Exception:
            return 'unknown'
    
    def _create_batch_prompt(self, text_content: str, questions: List[str], is_chinese: bool) -> str:
        """创建批量问题的提示词"""
        # 不对文本长度做任何限制，让模型处理完整文本
        # 如果文本超过模型的context window，那是模型本身的限制，不是评测程序的问题
        text_display = text_content
        
        if is_chinese:
            prompt = f"""请仔细阅读以下文本，然后依次回答所有问题。

文本内容：
{text_display}

请依次回答以下{len(questions)}个问题，每个问题只需给出一个数字答案：
"""
            for i, question in enumerate(questions, 1):
                prompt += f"{i}. {question}\n"
            
            prompt += f"""\n严格要求：
1. 绝对不要输出任何分析过程、解释或其他文字，只需按提问顺序给出数字答案，否则视为错误
2. 你必须回答全部{len(questions)}个问题，不能跳过任何一个
3. 必须输出恰好{len(questions)}个数字答案
4. 按顺序输出所有答案，用英文逗号分隔，不要有其他任何文字
5. 即使某个问题很难，也要给出最佳估计，不要留空

示例格式（{len(questions)}个答案）：{"5,23,0,17,8"[:2*len(questions)-1]}
你的答案："""
        else:
            prompt = f"""Please carefully read the following text and answer all questions in order.

Text Content:
{text_display}

Please answer the following {len(questions)} questions in order, providing only a number for each:
"""
            for i, question in enumerate(questions, 1):
                prompt += f"{i}. {question}\n"
            
            prompt += f"""\nSTRICT REQUIREMENTS:
1. NEVER output any analysis process, explanations or other text, only provide numeric answers in the order asked,otherwise it will be considered ERROR
2. You MUST answer ALL {len(questions)} questions, do NOT skip any
3. You MUST output exactly {len(questions)} numeric answers
4. Output all answers in order, separated by commas, with no other text
5. Even if a question is difficult, provide your best estimate, do NOT leave it blank

Example format ({len(questions)} answers): {"5,23,0,17,8"[:2*len(questions)-1]}
Your answer:"""
        
        return prompt
    
    def _parse_batch_response(self, response: str, num_questions: int) -> List[Optional[int]]:
        """解析批量响应，提取数字答案列表"""
        if not response:
            return [None] * num_questions
        
        # 不解析错误信息，避免从错误文本中提取数字
        if response.startswith("[MODEL_LIMIT_ERROR]") or response.startswith("[RATE_LIMIT_ERROR]"):
            return [None] * num_questions
        
        # 清理响应文本
        response = response.strip()
        original_response = response  # 保留原始文本用于think标签提取
        
        # 模式1：优先检查 think 标签格式
        try:
            # 匹配完整的 <think>...</think> 后面紧跟的数字列表
            think_pattern_full = re.compile(r"<think>.*?</think>\s*([\d,\s]+)", flags=re.IGNORECASE | re.DOTALL)
            think_match_full = think_pattern_full.search(original_response)
            if think_match_full:
                numbers_str = think_match_full.group(1)
                numbers = re.findall(r'\d+', numbers_str)
                if len(numbers) == num_questions:
                    results = []
                    for num_str in numbers:
                        try:
                            results.append(int(num_str))
                        except ValueError:
                            results.append(None)
                    return results
            
            # 匹配单独的 </think> 后面紧跟的数字列表
            think_pattern_end = re.compile(r"</think>\s*([\d,\s]+)", flags=re.IGNORECASE)
            think_match_end = think_pattern_end.search(original_response)
            if think_match_end:
                numbers_str = think_match_end.group(1)
                numbers = re.findall(r'\d+', numbers_str)
                if len(numbers) == num_questions:
                    results = []
                    for num_str in numbers:
                        try:
                            results.append(int(num_str))
                        except ValueError:
                            results.append(None)
                    return results
        except Exception:
            pass
        
        # 模式2：GLM 风格的 box 标记
        try:
            # 支持两种格式：begin_of_box 或 <|begin_of_box|>
            pattern = re.compile(r"(?:<\|)?begin_of_box(?:\|>)?\s*(.*?)\s*(?:<\|)?end_of_box(?:\|>)?", flags=re.IGNORECASE | re.DOTALL)
            boxes = list(pattern.finditer(response))
            if boxes:
                last_box_content = boxes[-1].group(1)
                numbers = re.findall(r'\d+', last_box_content)
                if len(numbers) == num_questions:
                    results = []
                    for num_str in numbers:
                        try:
                            results.append(int(num_str))
                        except ValueError:
                            results.append(None)
                    return results
        except Exception:
            pass
        
        # 模式3：纯数字的逗号分隔格式（最严格）
        pattern_pure = r'^[\d,\s]+$'
        if re.match(pattern_pure, response):
            numbers = re.findall(r'\d+', response)
            if len(numbers) == num_questions:
                results = []
                for num_str in numbers:
                    try:
                        results.append(int(num_str))
                    except ValueError:
                        results.append(None)
                return results
        
        # 不再使用宽泛的数字提取，避免从推理过程中提取数字
        # 解析失败，返回None列表
        return [None] * num_questions
    
    def evaluate_batch_tasks(self, tasks: List[TextCountingTask]) -> List[TextEvaluationResult]:
        """批量评测同一文档的多个任务"""
        if not tasks:
            return []
        
        # 确保所有任务来自同一文档
        document_id = tasks[0].document_id
        text_content = tasks[0].text_content
        
        # 检测语言
        text_is_chinese = TextLanguageDetector.is_chinese_text(text_content[:500])
        
        # 收集所有问题
        questions = [task.question for task in tasks]
        
        start_time = time.time()
        
        # 记录内存使用情况
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                memory_before = 0
        else:
            memory_before = 0
        
        try:
            self.logger.info(f"批量评测文档 {document_id} 的 {len(tasks)} 个问题")
            
            # 构建批量提示词
            user_prompt = self._create_batch_prompt(text_content, questions, text_is_chinese)
            
            # 批量评测专用系统提示词
            if text_is_chinese:
                system_prompt = (
                    "你是一个专业的文本分析计数专家。你必须回答所有问题，不能跳过任何一个。"
                    f"你将收到{len(questions)}个问题，必须给出恰好{len(questions)}个数字答案。"
                    "绝不能少于或多于要求的答案数量。"
                )
            else:
                system_prompt = (
                    "You are a professional text analysis and counting expert. You MUST answer ALL questions, do NOT skip any. "
                    f"You will receive {len(questions)} questions and MUST provide exactly {len(questions)} numeric answers. "
                    "NEVER provide fewer or more answers than required."
                )
            
            # 调用API
            max_retries = 3
            raw_response = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            for retry in range(max_retries):
                result = self.chat_bots.call(user_prompt, system_prompt=system_prompt, question_id=f"batch_{document_id}")
                
                if result is None:
                    self.logger.warning(f"批量API调用失败（第{retry+1}/{max_retries}次）")
                    continue
                
                raw_response, prompt_tokens, completion_tokens, batch_response_dict = result
                
                # 检查是否是模型限制错误或速率限制错误
                if isinstance(raw_response, str) and (raw_response.startswith("[MODEL_LIMIT_ERROR]") or 
                                                       raw_response.startswith("[RATE_LIMIT_ERROR]")):
                    self.logger.warning(f"批量评测遇到限制: {raw_response}")
                    # 模型限制或速率限制，不再重试，记录错误信息
                    break
                total_tokens = prompt_tokens + completion_tokens
                
                # 尝试解析响应
                parsed_answers = self._parse_batch_response(raw_response, len(questions))
                
                # 检查是否成功解析了所有答案
                valid_answers = sum(1 for ans in parsed_answers if ans is not None)
                if valid_answers == len(questions):  # 必须所有答案都有效
                    break
                else:
                    self.logger.warning(f"批量响应解析不完整（第{retry+1}/{max_retries}次），仅解析出 {valid_answers}/{len(questions)} 个答案")
                    if retry < max_retries - 1:
                        # 重试时使用更明确的提示
                        if text_is_chinese:
                            user_prompt = self._create_batch_prompt(text_content, questions, text_is_chinese)
                            user_prompt += f"\n\n注意：你必须给出恰好{len(questions)}个数字答案，用逗号分隔。"
                        else:
                            user_prompt = self._create_batch_prompt(text_content, questions, text_is_chinese)
                            user_prompt += f"\n\nNote: You must provide exactly {len(questions)} numeric answers, separated by commas."
            
            processing_time = time.time() - start_time
            
            # 计算内存使用
            if HAS_PSUTIL:
                try:
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_usage = memory_after - memory_before
                except Exception:
                    memory_usage = 0
            else:
                memory_usage = 0
            
            # 解析批量响应
            if raw_response and isinstance(raw_response, str) and (raw_response.startswith("[MODEL_LIMIT_ERROR]") or 
                                                                   raw_response.startswith("[RATE_LIMIT_ERROR]")):
                # 模型限制或速率限制错误，所有任务都标记为失败
                parsed_answers = [None] * len(questions)
            elif raw_response:
                parsed_answers = self._parse_batch_response(raw_response, len(questions))
            else:
                parsed_answers = [None] * len(questions)
            
            # 记录批量调用统计
            self.stats['batch_calls'] += 1
            
            # 为每个任务创建结果
            results = []
            avg_processing_time = processing_time / len(tasks) if tasks else 0
            avg_prompt_tokens = prompt_tokens / len(tasks) if tasks else 0
            avg_completion_tokens = completion_tokens / len(tasks) if tasks else 0
            avg_total_tokens = total_tokens / len(tasks) if tasks else 0
            
            for i, (task, answer) in enumerate(zip(tasks, parsed_answers)):
                if answer is not None:
                    is_correct = answer == task.ground_truth if task.ground_truth is not None else None
                    absolute_error = abs(answer - task.ground_truth) if task.ground_truth is not None else None
                    squared_error = (answer - task.ground_truth) ** 2 if task.ground_truth is not None else None
                    relative_error = absolute_error / max(task.ground_truth, 1) if task.ground_truth is not None else None
                else:
                    is_correct = False
                    absolute_error = float('inf')
                    squared_error = float('inf')
                    relative_error = float('inf')
                    answer = -1
                
                result = TextEvaluationResult(
                    task=task,
                    predicted_count=answer,
                    raw_response=raw_response if raw_response else "No response",  # 保存完整的原始回复
                    is_correct=is_correct if is_correct is not None else False,
                    absolute_error=absolute_error if absolute_error is not None else float('inf'),
                    relative_error=relative_error if relative_error is not None else float('inf'),
                    processing_time=avg_processing_time,
                    squared_error=squared_error,
                    memory_usage_mb=memory_usage / len(tasks),
                    prompt_tokens=int(avg_prompt_tokens),
                    completion_tokens=int(avg_completion_tokens),
                    total_tokens=int(avg_total_tokens),
                    response=batch_response_dict if 'batch_response_dict' in locals() else None
                )
                results.append(result)
            
            # 记录失败的任务到warning文件
            for result in results:
                if result.predicted_count == -1:
                    warning_key = (result.task.document_id, str(result.task.question_id))
                    if warning_key not in self.existing_warnings:
                        try:
                            warning_path = self.save_dir / 'warning_texts.txt'
                            safe_q = (result.task.question or '').replace('\t', ' ').replace('\n', ' ')
                            safe_resp = (result.raw_response or 'No response').replace('\t', ' ').replace('\n', ' ')  # 使用实际的原始回复
                            # 添加错误类型标记
                            if "[MODEL_LIMIT_ERROR]" in safe_resp:
                                safe_resp = f"[模型Context超限] {safe_resp}"
                            elif "[RATE_LIMIT_ERROR]" in safe_resp:
                                # 提取平台信息（Azure或其他）
                                if "TPM limit exceeded on Azure" in safe_resp:
                                    safe_resp = f"[TPM超Azure上限] {safe_resp}"
                                elif "TPM limit exceeded on API" in safe_resp:
                                    safe_resp = f"[TPM超API上限] {safe_resp}"
                                elif "TPM limit exceeded" in safe_resp:
                                    safe_resp = f"[TPM超限] {safe_resp}"
                                else:
                                    safe_resp = f"[Token速率超限] {safe_resp}"
                            with open(warning_path, 'a', encoding='utf-8') as wf:
                                wf.write(f"{result.task.document_id}\t{result.task.question_id}\t{safe_q}\t{safe_resp}\n")
                            # 添加到已记录集合中
                            self.existing_warnings.add(warning_key)
                        except Exception:
                            pass
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"批量评测失败: {str(e)}")
            
            # 返回所有任务的失败结果
            results = []
            for task in tasks:
                result = TextEvaluationResult(
                    task=task,
                    predicted_count=-1,
                    raw_response=f"Batch error: {str(e)}",
                    is_correct=False,
                    absolute_error=float('inf'),
                    relative_error=float('inf'),
                    processing_time=processing_time / len(tasks),
                    squared_error=float('inf'),
                    memory_usage_mb=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    error_message=str(e),
                    response=None
                )
                results.append(result)
            
            return results
    
    def evaluate_by_category(self, tasks: List[TextCountingTask], 
                          save_after_each: bool = True, 
                          save_intermediate: bool = False) -> List[TextEvaluationResult]:
        """
        按类别分组评测；每个类别完成后可立即保存该类别的完整结果文件。
        在批量模式下，同一文档的多个问题会合并为一次API调用。

        Args:
            tasks: 待评测任务
            save_after_each: 是否在每个类别完成后保存该类别的完整结果
            save_intermediate: 是否保存中间结果

        Returns:
            全部结果列表
        """
        if not tasks:
            return []

        # 分组
        tasks_by_category: Dict[str, List[TextCountingTask]] = {}
        for t in tasks:
            category = self._get_task_category(t)
            tasks_by_category.setdefault(category, []).append(t)

        all_results: List[TextEvaluationResult] = []
        
        # 初始化进度统计
        total_tasks = len(tasks)
        completed_tasks = 0
        start_time = time.time()
        self.logger.info(f"开始评测 {total_tasks} 个任务，分布在 {len(tasks_by_category)} 个类别中")
        
        # 逐类别评测
        for category, cat_tasks in tasks_by_category.items():
            self.logger.info(f"按类别评测开始: {category}（{len(cat_tasks)} 个任务）")
            category_results: List[TextEvaluationResult] = []

            if self.batch_mode:
                # 批量模式：按文档分组任务
                tasks_by_document: Dict[str, List[TextCountingTask]] = {}
                for task in cat_tasks:
                    doc_id = task.document_id
                    tasks_by_document.setdefault(doc_id, []).append(task)
                
                self.logger.info(f"[{category}] 批量模式：{len(tasks_by_document)} 个文档")
                
                # 获取已有的最大批次编号，用于续评时从正确的编号开始
                base_batch_num = self._get_max_batch_number(category)
                if base_batch_num > 0:
                    self.logger.info(f"[{category}] 检测到已有进度文件，最大批次编号: {base_batch_num}，将从此继续")
                
                last_saved_index = 0
                doc_count = 0
                
                for doc_id, doc_tasks in tasks_by_document.items():
                    try:
                        doc_count += 1
                        # 计算当前进度
                        current_percentage = completed_tasks / total_tasks * 100
                        elapsed_time = time.time() - start_time
                        avg_time_per_task = elapsed_time / completed_tasks if completed_tasks > 0 else 0
                        remaining_tasks = total_tasks - completed_tasks
                        eta_seconds = avg_time_per_task * remaining_tasks
                        
                        # 格式化剩余时间
                        if eta_seconds > 0:
                            eta_str = f", 预计剩余: {int(eta_seconds//60)}分{int(eta_seconds%60)}秒"
                        else:
                            eta_str = ""
                        
                        self.logger.info(f"[{category}] 批量评测文档 [{doc_count}/{len(tasks_by_document)}]: {doc_id} ({len(doc_tasks)} 个问题) | 总进度: {current_percentage:.1f}% ({completed_tasks}/{total_tasks}){eta_str}")
                        
                        # 批量评测同一文档的所有任务
                        batch_results = self.evaluate_batch_tasks(doc_tasks)
                        category_results.extend(batch_results)
                        
                        # 更新已完成任务数
                        completed_tasks += len(doc_tasks)
                        
                        # 中间结果保存（基于已有的最大批次编号递增）
                        if save_intermediate and (doc_count % 5 == 0 or doc_count == len(tasks_by_document)):
                            incremental = category_results[last_saved_index:]
                            if incremental:
                                # 计算当前批次编号：基础编号 + 当前文档计数
                                current_batch_num = base_batch_num + doc_count
                                self._save_intermediate_results(incremental, f"{category}_batch_{current_batch_num}")
                                last_saved_index = len(category_results)
                        
                        # 定期内存监控
                        if doc_count % 10 == 0:
                            gc.collect()
                            if HAS_PSUTIL:
                                process = psutil.Process()
                                memory_usage = process.memory_info().rss / 1024 / 1024
                                self._memory_samples.append(memory_usage)
                                if memory_usage > 2000:
                                    self.logger.warning(f"High memory usage detected: {memory_usage:.1f} MB")
                    
                    except Exception as e:
                        self.logger.error(f"[{category}] 批量评测文档 {doc_id} 失败: {str(e)}")
                        # 为该文档的所有任务创建错误结果
                        for task in doc_tasks:
                            error_result = TextEvaluationResult(
                                task=task,
                                predicted_count=-1,
                                raw_response=f"Batch error: {str(e)}",
                                is_correct=False,
                                absolute_error=float('inf'),
                                relative_error=float('inf'),
                                processing_time=0.0,
                                squared_error=float('inf'),
                                memory_usage_mb=0,
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                                error_message=str(e),
                                response=None
                            )
                            category_results.append(error_result)
            else:
                # 单任务模式：逐个评测
                # 获取已有的最大批次编号，用于续评时从正确的编号开始
                base_batch_num = self._get_max_batch_number(category)
                if base_batch_num > 0:
                    self.logger.info(f"[{category}] 检测到已有进度文件，最大批次编号: {base_batch_num}，将从此继续")
                
                last_saved_index = 0
                for i, task in enumerate(cat_tasks):
                    try:
                        # 计算当前进度
                        current_percentage = completed_tasks / total_tasks * 100
                        elapsed_time = time.time() - start_time
                        avg_time_per_task = elapsed_time / completed_tasks if completed_tasks > 0 else 0
                        remaining_tasks = total_tasks - completed_tasks
                        eta_seconds = avg_time_per_task * remaining_tasks
                        
                        # 格式化剩余时间
                        if eta_seconds > 0:
                            eta_str = f", 预计剩余: {int(eta_seconds//60)}分{int(eta_seconds%60)}秒"
                        else:
                            eta_str = ""
                        
                        self.logger.info(f"[{category}] 评测任务 [{i+1}/{len(cat_tasks)}]: {task.question[:50]}... | 总进度: {current_percentage:.1f}% ({completed_tasks}/{total_tasks}){eta_str}")
                        result = self.evaluate_single_task(task)
                        category_results.append(result)
                        self.stats['single_calls'] += 1
                        
                        # 更新已完成任务数
                        completed_tasks += 1

                        # 类别内按固定频率保存增量中间结果（默认关闭）
                        if save_intermediate and ((i + 1) % 10 == 0 or (i + 1) == len(cat_tasks)):
                            incremental = category_results[last_saved_index:]
                            if incremental:
                                # 计算当前批次编号：基础编号 + 当前任务序号
                                current_batch_num = base_batch_num + (i + 1)
                                self._save_intermediate_results(incremental, f"{category}_{current_batch_num}")
                                last_saved_index = len(category_results)
                        
                        # 定期内存监控和垃圾收集（按类别内进度）
                        if (i + 1) % 20 == 0:
                            gc.collect()
                            if HAS_PSUTIL:
                                process = psutil.Process()
                                memory_usage = process.memory_info().rss / 1024 / 1024
                                self._memory_samples.append(memory_usage)
                                if memory_usage > 2000:  # 超过2GB时警告
                                    self.logger.warning(f"High memory usage detected: {memory_usage:.1f} MB")

                    except Exception as e:
                        self.logger.error(f"[{category}] 任务失败: {str(e)}")
                        # 记录错误结果
                        error_result = TextEvaluationResult(
                            task=task,
                            predicted_count=-1,
                            raw_response=f"Error: {str(e)}",
                            is_correct=False,
                            absolute_error=float('inf'),
                            relative_error=float('inf'),
                            processing_time=0.0,
                            squared_error=float('inf'),
                            memory_usage_mb=0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            error_message=str(e),
                            response=None
                        )
                        category_results.append(error_result)

            # 类别完成后保存完整类别结果
            if save_after_each and category_results:
                self._save_category_results(category_results, category)
                self.logger.info(f"按类别保存完成: {category}")

            all_results.extend(category_results)
            
            # 显示类别完成进度
            category_percentage = completed_tasks / total_tasks * 100
            elapsed_time = time.time() - start_time
            self.logger.info(f"类别 {category} 完成 | 总进度: {category_percentage:.1f}% ({completed_tasks}/{total_tasks}), 已用时: {int(elapsed_time//60)}分{int(elapsed_time%60)}秒")

        # 显示最终完成统计
        total_time = time.time() - start_time
        avg_time_per_task = total_time / total_tasks if total_tasks > 0 else 0
        self.logger.info(f"按类别评测完成，总结果数: {len(all_results)}")
        self.logger.info(f"总耗时: {int(total_time//60)}分{int(total_time%60)}秒, 平均每任务: {avg_time_per_task:.2f}秒")
        return all_results
    
    def generate_report(self, results: List[TextEvaluationResult]) -> Dict[str, Any]:
        """
        生成文本评测报告（与图像评测器保持一致）
        
        Args:
            results: 结果列表
            
        Returns:
            评测报告
        """
        if not results:
            return {"error": "没有结果数据"}

        total_tasks = len(results)
        successful_extractions = sum(1 for r in results if r.predicted_count is not None and r.predicted_count != -1)
        total_tokens_used = sum((r.total_tokens or 0) for r in results)

        # 计算准确率（只考虑有ground truth的任务）
        tasks_with_gt = [r for r in results if r.task.ground_truth is not None]
        correct_predictions = sum(1 for r in tasks_with_gt if r.is_correct is True)
        
        # 新增：总体准确率（包含所有任务，提取失败算作错误）
        overall_accuracy = correct_predictions / len(tasks_with_gt) if tasks_with_gt else 0
        
        # 有效准确率（只考虑成功提取的任务）
        valid_results_with_gt = [r for r in tasks_with_gt if r.predicted_count != -1]
        valid_accuracy = sum(1 for r in valid_results_with_gt if r.is_correct is True) / len(valid_results_with_gt) if valid_results_with_gt else 0
        
        # 保持向后兼容
        accuracy = overall_accuracy

        # 计算平均处理时间
        avg_processing_time = sum(r.processing_time for r in results) / total_tasks if total_tasks > 0 else 0

        # 按类别分组统计
        category_stats = {}
        for result in results:
            category = self._get_task_category(result.task)
            if category not in category_stats:
                category_stats[category] = {
                    'total_questions': 0,
                    'total_documents': set(),
                    'successful': 0,
                    'correct': 0,
                    'with_gt': 0
                }

            category_stats[category]['total_questions'] += 1
            category_stats[category]['total_documents'].add(result.task.document_id)
            if result.predicted_count is not None and result.predicted_count != -1:
                category_stats[category]['successful'] += 1
            if result.task.ground_truth is not None:
                category_stats[category]['with_gt'] += 1
                if result.is_correct is True:
                    category_stats[category]['correct'] += 1

        # 计算各类别准确率和转换 total_documents 为数量
        for category in category_stats:
            stats = category_stats[category]
            stats['total_documents'] = len(stats['total_documents'])  # 转换为数量
            stats['total'] = stats['total_questions']  # 保持向后兼容
            stats['accuracy'] = stats['correct'] / stats['with_gt'] if stats['with_gt'] > 0 else 0
            stats['extraction_rate'] = stats['successful'] / stats['total_questions'] if stats['total_questions'] > 0 else 0

        # 添加批量模式统计（如果启用）
        batch_stats = {}
        if hasattr(self, 'batch_mode') and self.batch_mode:
            batch_stats = {
                "batch_mode_enabled": True,
                "batch_api_calls": self.stats.get('batch_calls', 0),
                "single_api_calls": self.stats.get('single_calls', 0),
                "total_api_calls": self.stats.get('batch_calls', 0) + self.stats.get('single_calls', 0)
            }
        
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_extractions": successful_extractions,
                "extraction_rate": successful_extractions / total_tasks if total_tasks > 0 else 0,
                "tasks_with_ground_truth": len(tasks_with_gt),
                "correct_predictions": correct_predictions,
                "overall_accuracy": overall_accuracy,  # 总体准确率
                "valid_accuracy": valid_accuracy,      # 有效准确率
                "accuracy": accuracy,                  # 保持向后兼容，使用总体准确率
                "average_processing_time": avg_processing_time,
                "total_tokens": total_tokens_used,
                **batch_stats
            },
            "by_category": category_stats,
            "timestamp": datetime.now().isoformat()
        }

        return report
    
    def evaluate_tasks(self, tasks: List[TextCountingTask], 
                      save_results: bool = True) -> List[TextEvaluationResult]:
        """
        批量评测文本计数任务
        
        Args:
            tasks: 文本计数任务列表
            save_results: 是否保存结果
            
        Returns:
            List[TextEvaluationResult]: 评测结果列表
        """
        self.stats['total_tasks'] = len(tasks)
        self.stats['start_time'] = datetime.now()
        
        self.logger.info(f"Starting evaluation of {len(tasks)} text counting tasks")
        
        results = []
        
        for i, task in enumerate(tasks):
            try:
                self.logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task.question[:50]}...")
                
                result = self.evaluate_single_task(task)
                results.append(result)
                
                if result.predicted_count != -1:
                    self.stats['completed_tasks'] += 1
                else:
                    self.stats['failed_tasks'] += 1
                
                # 内存清理和性能监控
                if (i + 1) % 50 == 0:
                    gc.collect()
                    if HAS_PSUTIL:
                        process = psutil.Process()
                        memory_usage = process.memory_info().rss / 1024 / 1024
                        cpu_percent = process.cpu_percent(interval=None)
                        self._memory_samples.append(memory_usage)
                        self.logger.info(f"Progress: {i+1}/{len(tasks)} - Memory: {memory_usage:.1f} MB, CPU: {cpu_percent:.1f}%")
                    else:
                        self.logger.info(f"Progress: {i+1}/{len(tasks)} tasks completed")
                
                # 定期保存中间结果
                if save_results and (i + 1) % 100 == 0:
                    self._save_intermediate_results(results, i + 1)
                
            except KeyboardInterrupt:
                self.logger.info("Evaluation interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in task {i+1}: {str(e)}")
                self.stats['failed_tasks'] += 1
                continue
        
        self.stats['end_time'] = datetime.now()
        
        # 计算性能统计
        if self._memory_samples:
            self.stats['peak_memory_mb'] = max(self._memory_samples)
            self.stats['average_memory_mb'] = sum(self._memory_samples) / len(self._memory_samples)
        
        # 计算总体统计
        if results:
            total_tokens = sum(r.total_tokens for r in results if r.total_tokens)
            total_time = sum(r.processing_time for r in results)
            self.stats['total_tokens'] = total_tokens
            self.stats['average_processing_time'] = total_time / len(results) if results else 0
        
        if save_results:
            self._save_final_results(results)
        
        self._print_summary(results)
        
        return results
    
    def _get_max_batch_number(self, category: str) -> int:
        """
        获取指定类别已有的最大批次编号
        
        Args:
            category: 类别名称
            
        Returns:
            最大批次编号，如果没有找到则返回0
        """
        category_dir = self.save_dir / f"{category}_results"
        if not category_dir.exists():
            return 0
        
        max_batch = 0
        # 扫描所有batch_progress文件
        for file_path in category_dir.glob(f"{category}_batch_progress_{category}_batch_*.json"):
            try:
                # 从文件名中提取批次编号，如：code_batch_progress_code_batch_75.json -> 75
                filename = file_path.stem
                # 分割文件名获取最后的数字部分
                parts = filename.split('_')
                if parts and parts[-1].isdigit():
                    batch_num = int(parts[-1])
                    max_batch = max(max_batch, batch_num)
            except (ValueError, IndexError):
                continue
        
        return max_batch
    
    def _save_intermediate_results(self, results: List[TextEvaluationResult], batch_id: str):
        """
        保存中间结果增量（仅写入本批新增的结果）。
        在当前运行目录下为每个类别创建子文件夹，并将本批增量写入
        "{category}_batch_progress_{batch_id}.json"。
        """
        # 按类别分组当前这批增量结果
        results_by_category = {}
        for result in results:
            category = self._get_task_category(result.task)
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        # 写出增量文件
        results_root = self.save_dir
        results_root.mkdir(exist_ok=True, parents=True)
        for category, category_results in results_by_category.items():
            category_dir = results_root / f"{category}_results"
            category_dir.mkdir(exist_ok=True)
            category_filename = f"{category}_batch_progress_{batch_id}.json"
            save_path = category_dir / category_filename
            results_data = [result.to_dict() for result in category_results]
            writeJson(results_data, str(save_path))
            self.logger.info(f"Saved intermediate {category} results to: {save_path}")
    
    def _save_category_results(self, results: List[TextEvaluationResult], category: str):
        """
        保存类别结果（按类别创建文件夹）
        会自动合并所有batch_progress文件，确保断点续评后的完整性
        """
        # 为每个类别创建文件夹
        category_dir = self.save_dir / f"{category}_results"
        category_dir.mkdir(exist_ok=True)
        
        # 检查是否存在batch进度文件（断点续评的标志）
        batch_files = sorted(category_dir.glob(f"{category}_batch_progress_*.json"))
        
        if len(batch_files) > 0:
            # 存在batch文件，需要合并所有历史数据
            self.logger.info(f"检测到 {len(batch_files)} 个batch文件，将合并所有数据...")
            
            # 用于去重的字典：key = (document_id, question_id)
            merged_results = {}
            
            # 1. 先加载所有batch文件的数据
            for batch_file in batch_files:
                try:
                    batch_data = readJson(str(batch_file))
                    for item in batch_data:
                        # 提取唯一键
                        task_info = item.get('task', {})
                        doc_id = task_info.get('document_id', '')
                        q_id = task_info.get('question_id', '')
                        key = (doc_id, q_id)
                        # 后面的覆盖前面的（保留最新的结果）
                        merged_results[key] = item
                except Exception as e:
                    self.logger.warning(f"读取batch文件失败 {batch_file}: {e}")
            
            # 2. 再加入当前结果（覆盖之前的重复项）
            for result in results:
                result_dict = result.to_dict()
                task_info = result_dict.get('task', {})
                doc_id = task_info.get('document_id', '')
                q_id = task_info.get('question_id', '')
                key = (doc_id, q_id)
                merged_results[key] = result_dict
            
            # 3. 保存合并后的完整结果
            results_data = list(merged_results.values())
            self.logger.info(f"合并完成：{len(results_data)} 个任务（去重后）")
        else:
            # 没有batch文件，直接保存当前结果
            results_data = [result.to_dict() for result in results]
            self.logger.info(f"首次保存：{len(results_data)} 个任务")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{category}_results_{timestamp}.json"
        filepath = category_dir / filename
        
        writeJson(results_data, str(filepath))
        self.logger.info(f"Saved {category} category results to: {filepath}")
    
    def save_results(self, results: List[TextEvaluationResult], filename: str = None):
        """
        保存评测结果，按类别创建文件夹

        Args:
            results: 结果列表
            filename: 文件名，如果为None则自动生成
        """
        # 按类别分组结果
        results_by_category = {}
        for result in results:
            category = self._get_task_category(result.task)
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []

        # 为每个类别创建文件夹并保存结果
        for category, category_results in results_by_category.items():
            # 创建类别文件夹
            category_dir = self.save_dir / f"{category}_results"
            category_dir.mkdir(exist_ok=True)

            # 生成文件名
            if filename is None:
                category_filename = f"{category}_results_{timestamp}.json"
            else:
                category_filename = f"{category}_{filename}"

            save_path = category_dir / category_filename

            # 保存完整的结果数据
            results_data = [result.to_dict() for result in category_results]
            writeJson(results_data, str(save_path))

            self.logger.info(f"{category} 类别结果已保存到: {save_path}")
            saved_paths.append(save_path)

        return saved_paths
    
    def _save_final_results(self, results: List[TextEvaluationResult]):
        """保存最终结果（使用新的按类别组织的方式）"""
        # 使用新的 save_results 方法
        saved_paths = self.save_results(results)
        
        # 保存统计信息
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_file = self.save_dir / f"text_evaluation_stats_{timestamp}.json"
        writeJson(self.stats, str(stats_file))
        
        self.logger.info(f"Saved statistics to: {stats_file}")
        return saved_paths
    
    def _print_summary(self, results: List[TextEvaluationResult]):
        """打印评测摘要"""
        if not results:
            self.logger.info("No results to summarize")
            return
        
        valid_results = [r for r in results if r.predicted_count != -1]
        
        if not valid_results:
            self.logger.info("No valid results to analyze")
            return
        
        # 基本统计
        total_tasks = len(results)
        valid_tasks = len(valid_results)
        
        # 计算准确率（只考虑有ground truth的任务）
        tasks_with_gt = [r for r in results if r.task.ground_truth is not None]
        correct_predictions = sum(1 for r in tasks_with_gt if r.is_correct is True)
        
        # 总体准确率（包含所有任务，提取失败算作错误）
        overall_accuracy = correct_predictions / len(tasks_with_gt) if tasks_with_gt else 0
        
        # 有效准确率（只考虑成功提取的任务）
        valid_results_with_gt = [r for r in tasks_with_gt if r.predicted_count != -1]
        valid_correct = sum(1 for r in valid_results_with_gt if r.is_correct is True)
        valid_accuracy = valid_correct / len(valid_results_with_gt) if valid_results_with_gt else 0
        
        # 保持向后兼容
        correct_tasks = correct_predictions
        accuracy = overall_accuracy
        
        # 误差统计
        absolute_errors = [r.absolute_error for r in valid_results if r.absolute_error != float('inf')]
        if absolute_errors:
            mae = np.mean(absolute_errors)
            rmse = np.sqrt(np.mean([e**2 for e in absolute_errors]))
        else:
            mae = rmse = float('inf')
        
        # 按难度级别统计
        level_stats = {}
        for result in valid_results:
            level = result.task.level
            if level not in level_stats:
                level_stats[level] = {'total': 0, 'correct': 0}
            level_stats[level]['total'] += 1
            if result.is_correct:
                level_stats[level]['correct'] += 1
        
        # 打印摘要
        print("\n" + "="*60)
        print("TEXT COUNTING EVALUATION SUMMARY")
        print("="*60)
        print(f"Total tasks: {total_tasks}")
        print(f"Valid predictions: {valid_tasks}")
        print(f"Failed predictions: {total_tasks - valid_tasks}")
        print(f"Correct predictions: {correct_tasks}")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")  # 总体准确率
        print(f"Valid Accuracy: {valid_accuracy:.3f}")      # 有效准确率
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.3f}")
        
        # 性能统计
        if self.stats.get('peak_memory_mb', 0) > 0:
            print(f"\nPerformance Statistics:")
            print(f"Peak Memory Usage: {self.stats['peak_memory_mb']:.1f} MB")
            print(f"Average Memory Usage: {self.stats['average_memory_mb']:.1f} MB")
        if self.stats.get('total_tokens', 0) > 0:
            print(f"Total Tokens Used: {self.stats['total_tokens']}")
        if self.stats.get('average_processing_time', 0) > 0:
            print(f"Average Processing Time: {self.stats['average_processing_time']:.2f} seconds")
        
        # 批量模式统计
        if hasattr(self, 'batch_mode') and self.batch_mode:
            print(f"\nBatch Mode Statistics:")
            print(f"Batch API Calls: {self.stats.get('batch_calls', 0)}")
            print(f"Single API Calls: {self.stats.get('single_calls', 0)}")
            total_api_calls = self.stats.get('batch_calls', 0) + self.stats.get('single_calls', 0)
            print(f"Total API Calls: {total_api_calls}")
            if total_api_calls > 0 and total_tasks > 0:
                print(f"API Efficiency: {total_tasks / total_api_calls:.2f} tasks per API call")
        
        print(f"\nAccuracy by difficulty level:")
        for level, stats in level_stats.items():
            level_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {level}: {level_accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            print(f"\nEvaluation duration: {duration}")
        
        print("="*60)


def _text_task_category(task) -> str:
    """从文本任务推断类别"""
    try:
        return Path(getattr(task, 'document_id', '')).parent.name or 'unknown'
    except Exception:
        return 'unknown'
