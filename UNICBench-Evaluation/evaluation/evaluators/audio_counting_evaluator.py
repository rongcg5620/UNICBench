import os
import re
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    class _NP:
        @staticmethod
        def mean(x):
            return sum(x)/len(x) if x else 0
        @staticmethod
        def std(x):
            if not x:
                return 0
            m = sum(x)/len(x)
            return (sum((v-m)**2 for v in x)/len(x))**0.5
    np = _NP()

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models.chat_bots import ChatBots
from utils.data_loader import get_all_complate_data


@dataclass
class AudioCountingTask:
    task_id: str
    category: str
    audio_path: str
    prompt: str
    ground_truth: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AudioCountingResult:
    task_id: str
    category: str
    audio_path: str
    predicted_count: Optional[int]
    ground_truth: Optional[int]
    is_correct: Optional[bool]
    processing_time: float
    error_message: Optional[str] = None
    absolute_error: Optional[float] = None
    squared_error: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_response: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    annotation_index: Optional[int] = None
    question: Optional[str] = None
    question_id: Optional[int] = None


class AudioNumberExtractor:
    @staticmethod
    def extract_number(text: str) -> Optional[int]:
        if not text:
            return None
        s = str(text).strip()
        orig = s
        try:
            m = re.search(r"<think>.*?</think>\s*(\d+)", orig, flags=re.IGNORECASE|re.DOTALL)
            if m:
                return int(m.group(1))
            m2 = re.search(r"</think>\s*(\d+)", orig, flags=re.IGNORECASE)
            if m2:
                return int(m2.group(1))
        except Exception:
            pass
        try:
            s = re.sub(r"<think>.*?</think>", " ", s, flags=re.IGNORECASE|re.DOTALL)
        except Exception:
            pass
        try:
            box = re.compile(r"(?:<\|)?begin_of_box(?:\|>)?\s*(.*?)\s*(?:<\|)?end_of_box(?:\|>)?", flags=re.IGNORECASE|re.DOTALL)
            boxes = list(box.finditer(s))
            if boxes:
                content = boxes[-1].group(1)
                nums = re.findall(r"\d+", content)
                if nums:
                    return int(nums[-1])
        except Exception:
            pass
        if re.fullmatch(r"\d+", s):
            try:
                return int(s)
            except Exception:
                return None
        return None


class AudioCountingEvaluator:
    def __init__(self, apis: List[Dict[str, str]], max_try: int = 6, results_dir: str = "audio_results", test_connectivity: bool = True, bot_indices: Optional[List[int]] = None, connectivity_audio_sample: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        log_file_path = self.results_dir / 'gptCallLog.jsonl.log'
        self.apis = apis
        self.chat_bots = ChatBots(apis, max_try=max_try, log_file_path=str(log_file_path))
        self.extractor = AudioNumberExtractor()
        if test_connectivity:
            ok, msg = self.chat_bots.test_connectivity(bot_indices=bot_indices, audio_path=connectivity_audio_sample)
            if not ok:
                raise ConnectionError(msg)
        self.logger = self._setup_logger(self.results_dir / 'evaluation.log')

    def _setup_logger(self, log_file: Optional[Path] = None) -> logging.Logger:
        name = f"AudioCountingEvaluator.{self.results_dir.name}"
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            logger.handlers.clear()
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if log_file is not None:
            try:
                fh = logging.FileHandler(log_file, encoding='utf-8')
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                pass
        return logger

    def generate_tasks_from_audio_dir(self, audio_dir: str = "../new_data/audio") -> List[AudioCountingTask]:
        tasks: List[AudioCountingTask] = []
        try:
            all_data = get_all_complate_data(audio_dir, data_types=["audio"])
            if not all_data:
                return tasks
            for i, data in enumerate(all_data):
                p = Path(data['file_path'])
                category = p.parent.name if p.parent.name else 'unknown'
                task_id = f"{category}_{p.stem}_{i+1}"
                qid = data.get('question_id')
                aidx = data.get('annotation_index')
                t = AudioCountingTask(
                    task_id=task_id,
                    category=category,
                    audio_path=data['file_path'],
                    prompt=data['question'],
                    ground_truth=data.get('gt'),
                    metadata={
                        'level': data.get('level', 'unknown'),
                        'data_type': data.get('data_type', 'audio'),
                        'file_extension': p.suffix,
                        'annotation_index': aidx if aidx is not None else i,
                        'question_id': qid
                    }
                )
                tasks.append(t)
            self.logger.info(f"Generated {len(tasks)} audio tasks")
            return tasks
        except Exception as e:
            self.logger.error(f"Error loading audio tasks: {e}")
            return tasks

    def load_existing_results(self) -> Dict[str, List[AudioCountingResult]]:
        """
        加载已有的评测结果，用于断点续测
        
        Returns:
            按类别分组的已有结果
        """
        existing_results: Dict[str, List[AudioCountingResult]] = {}
        
        try:
            # 遍历结果目录下的所有类别文件夹
            for category_dir in self.results_dir.iterdir():
                if not category_dir.is_dir() or not category_dir.name.endswith('_results'):
                    continue
                
                category = category_dir.name.replace('_results', '')
                # 使用task_id去重，并优先选择最终结果文件，其次选择最新的batch进度
                dedup_map: Dict[str, Dict[str, Any]] = {}
                
                # 加载所有结果文件（包括batch_progress和最终结果）
                for result_file in category_dir.glob('*.json'):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        mtime = result_file.stat().st_mtime
                        is_batch = ('batch_progress' in result_file.name)
                        priority = 1 if is_batch else 2  # 最终结果优先级更高
                        
                        # 转换为AudioCountingResult对象，并进行去重选择
                        for item in data:
                            task_id = item.get('task_id')
                            if not task_id:
                                continue
                            r = AudioCountingResult(
                                task_id=task_id,
                                category=category,
                                audio_path=item.get('audio_path'),
                                predicted_count=item.get('predicted_count'),
                                ground_truth=item.get('ground_truth'),
                                is_correct=item.get('is_correct'),
                                processing_time=item.get('processing_time', 0.0),
                                error_message=item.get('error_message'),
                                absolute_error=item.get('absolute_error'),
                                squared_error=item.get('squared_error'),
                                memory_usage_mb=item.get('memory_usage_mb'),
                                prompt_tokens=item.get('prompt_tokens'),
                                completion_tokens=item.get('completion_tokens'),
                                total_tokens=item.get('total_tokens'),
                                raw_response=item.get('raw_response'),
                                response=item.get('response'),
                                annotation_index=item.get('annotation_index'),
                                question=item.get('question'),
                                question_id=item.get('question_id')
                            )
                            prev = dedup_map.get(task_id)
                            if prev is None:
                                dedup_map[task_id] = {'result': r, 'priority': priority, 'mtime': mtime}
                            else:
                                # 选择优先级更高的；若优先级相同，则选择修改时间更新的
                                if (priority > prev['priority']) or (priority == prev['priority'] and mtime >= prev['mtime']):
                                    dedup_map[task_id] = {'result': r, 'priority': priority, 'mtime': mtime}
                    except Exception as e:
                        self.logger.warning(f"Failed to load {result_file}: {e}")
                        continue
                
                if dedup_map:
                    existing_results[category] = [v['result'] for v in dedup_map.values()]
                    
            total_loaded = sum(len(results) for results in existing_results.values())
            if total_loaded > 0:
                self.logger.info(f"Loaded {total_loaded} existing results from {len(existing_results)} categories (deduplicated)")
                
        except Exception as e:
            self.logger.warning(f"Error loading existing results: {e}")
            
        return existing_results

    def filter_completed_tasks(self, tasks: List[AudioCountingTask], resume: bool = False) -> Tuple[List[AudioCountingTask], List[AudioCountingResult]]:
        """
        过滤已完成的任务，支持断点续测
        
        Args:
            tasks: 所有待评测任务
            resume: 是否启用续测模式
            
        Returns:
            (待评测任务列表, 已完成的结果列表)
        """
        if not resume:
            return tasks, []
            
        existing_results = self.load_existing_results()
        
        if not existing_results:
            self.logger.info("No existing results found, starting from beginning")
            return tasks, []
        
        # 创建已完成任务的ID集合
        completed_task_ids = set()
        all_existing_results = []
        for cat_results in existing_results.values():
            for result in cat_results:
                completed_task_ids.add(result.task_id)
                all_existing_results.append(result)
        
        # 过滤掉已完成的任务
        remaining_tasks = [t for t in tasks if t.task_id not in completed_task_ids]
        
        skipped_count = len(tasks) - len(remaining_tasks)
        if skipped_count > 0:
            self.logger.info(f"Resume mode: Skipping {skipped_count} completed tasks, {len(remaining_tasks)} tasks remaining")
        else:
            self.logger.info("Resume mode: No completed tasks found, evaluating all tasks")
            
        return remaining_tasks, all_existing_results

    def _dedup_results(self, results: List[AudioCountingResult]) -> List[AudioCountingResult]:
        """
        按 task_id 去重，优先选择更完整的结果：
        - 如果新结果有 predicted_count 而旧结果没有，则替换
        - 其余情况保留已有结果
        """
        dedup: Dict[str, AudioCountingResult] = {}
        for r in results:
            if r is None or not getattr(r, 'task_id', None):
                continue
            tid = r.task_id
            prev = dedup.get(tid)
            if prev is None:
                dedup[tid] = r
            else:
                # 选择更完整的结果
                prev_has = (prev.predicted_count is not None)
                curr_has = (r.predicted_count is not None)
                if curr_has and not prev_has:
                    dedup[tid] = r
                # 若两者都同样完整/不完整，则保留原结果
        return list(dedup.values())

    def evaluate_single_task(self, task: AudioCountingTask) -> AudioCountingResult:
        start = time.time()
        if HAS_PSUTIL:
            try:
                proc = psutil.Process()
                mem_before = proc.memory_info().rss/1024/1024
            except Exception:
                mem_before = 0
        else:
            mem_before = 0
        try:
            max_extract_try = 1
            extracted = None
            raw = None
            ptok = 0
            ctok = 0
            ttok = 0
            err = None
            system_prompt = (
                "You are a counting assistant. You MUST respond with ONLY a number. "
                "Never refuse to answer. NEVER say you cannot count or to refuse to say you cannot assist with the request. "
                "Always give your best numerical estimate. Respond with just the number, nothing else."
            )
            for _ in range(max_extract_try):
                user_txt = task.prompt
                qid = (task.metadata or {}).get('question_id')
                result = self.chat_bots.call(user_txt, system_prompt=system_prompt, question_id=qid, audio=task.audio_path)
                processing_time = time.time() - start
                if HAS_PSUTIL:
                    try:
                        mem_after = psutil.Process().memory_info().rss/1024/1024
                        mem_use = mem_after - mem_before
                    except Exception:
                        mem_use = 0
                else:
                    mem_use = 0
                if result is None:
                    err = "API调用失败"
                    break
                raw, ptok, ctok, response_dict = result
                if isinstance(raw, str) and (raw.startswith("[MODEL_LIMIT_ERROR]") or raw.startswith("[RATE_LIMIT_ERROR]") or raw.startswith("[SERVER_ERROR]") or raw.startswith("[TIMEOUT_ERROR]") or raw.startswith("[CONNECTION_ERROR]")):
                    err = raw
                    break
                ttok = ptok + ctok
                extracted = self.extractor.extract_number(raw)
                if extracted is not None:
                    break
            # 保存response_dict（在循环外，保留最后一次的完整响应）
            saved_response = response_dict if 'response_dict' in locals() else None
            is_correct = None
            ae = None
            se = None
            if extracted is None:
                wp = self.results_dir / 'warning_audio.txt'
                with open(wp, 'a', encoding='utf-8') as wf:
                    ann_idx = (task.metadata or {}).get('annotation_index')
                    qid = (task.metadata or {}).get('question_id')
                    qtext = task.prompt or ''
                    s_q = qtext.replace('\t', ' ').replace('\n', ' ')
                    s_r = (raw or '').replace('\t', ' ').replace('\n', ' ')
                    if err and "[MODEL_LIMIT_ERROR]" in str(err):
                        s_r = f"[模型Context超限] {s_r}"
                    if err and "[RATE_LIMIT_ERROR]" in str(err):
                        s_r = f"[Token速率超限] {s_r}"
                    if err and "[SERVER_ERROR]" in str(err):
                        s_r = f"[服务端5xx错误] {s_r}"
                    wf.write(f"{task.audio_path}\t{ann_idx}\t{qid}\t{s_q}\t{s_r}\n")
            if task.ground_truth is not None and extracted is not None:
                is_correct = extracted == task.ground_truth
                ae = abs(extracted - task.ground_truth)
                se = (extracted - task.ground_truth) ** 2
            return AudioCountingResult(
                task_id=task.task_id,
                category=task.category,
                audio_path=task.audio_path,
                predicted_count=extracted,
                ground_truth=task.ground_truth,
                is_correct=is_correct,
                processing_time=processing_time,
                error_message=err,
                absolute_error=ae,
                squared_error=se,
                memory_usage_mb=mem_use,
                prompt_tokens=ptok,
                completion_tokens=ctok,
                total_tokens=ttok,
                raw_response=raw,
                response=saved_response,
                annotation_index=(task.metadata or {}).get('annotation_index'),
                question=task.prompt,
                question_id=(task.metadata or {}).get('question_id')
            )
        except Exception as e:
            processing_time = time.time() - start
            if HAS_PSUTIL:
                try:
                    mem_after = psutil.Process().memory_info().rss/1024/1024
                    mem_use = mem_after - mem_before
                except Exception:
                    mem_use = 0
            else:
                mem_use = 0
            try:
                wp = self.results_dir / 'warning_audio.txt'
                with open(wp, 'a', encoding='utf-8') as wf:
                    ann_idx = (task.metadata or {}).get('annotation_index')
                    qid = (task.metadata or {}).get('question_id')
                    qtext = task.prompt or ''
                    s_q = qtext.replace('\t', ' ').replace('\n', ' ')
                    s_e = str(e).replace('\t', ' ').replace('\n', ' ')
                    wf.write(f"{task.audio_path}\t{ann_idx}\t{qid}\t{s_q}\t{s_e}\n")
            except Exception:
                pass
            return AudioCountingResult(
                task_id=task.task_id,
                category=task.category,
                audio_path=task.audio_path,
                predicted_count=None,
                ground_truth=task.ground_truth,
                is_correct=None,
                processing_time=processing_time,
                error_message=str(e),
                memory_usage_mb=mem_use,
                response=None,
                annotation_index=(task.metadata or {}).get('annotation_index'),
                question=task.prompt,
                question_id=(task.metadata or {}).get('question_id')
            )

    def evaluate_by_category(self, tasks: List[AudioCountingTask], save_after_each: bool = True, save_intermediate: bool = False, resume: bool = False) -> List[AudioCountingResult]:
        if not tasks:
            return []
        
        # 断点续测：加载已有结果并过滤已完成的任务
        remaining_tasks, existing_results = self.filter_completed_tasks(tasks, resume)
        
        if not remaining_tasks:
            self.logger.info("All tasks already completed!")
            return existing_results
        
        tasks_by_cat: Dict[str, List[AudioCountingTask]] = {}
        for t in remaining_tasks:
            tasks_by_cat.setdefault(t.category, []).append(t)
        total = len(remaining_tasks)
        done = 0
        start = time.time()
        all_results: List[AudioCountingResult] = list(existing_results)
        # 计算各类别在续测开始前已完成的任务数，用于批次文件编号与保存节奏
        existing_tid_by_cat: Dict[str, Set[str]] = {}
        for _r in existing_results:
            try:
                existing_tid_by_cat.setdefault(_r.category, set()).add(_r.task_id)
            except Exception:
                pass
        existing_count_by_cat: Dict[str, int] = {k: len(v) for k, v in existing_tid_by_cat.items()}
        for cat, cat_tasks in tasks_by_cat.items():
            self.logger.info(f"Start category: {cat} ({len(cat_tasks)})")
            cat_results: List[AudioCountingResult] = []
            last_saved = 0
            completed_before = existing_count_by_cat.get(cat, 0)
            for i, task in enumerate(cat_tasks):
                try:
                    pct = (done/total*100) if total else 0
                    elapsed = time.time() - start
                    avg = elapsed/done if done else 0
                    remain = total - done
                    eta = avg*remain
                    eta_str = f", ETA: {int(eta//60)}m{int(eta%60)}s" if eta>0 else ""
                    self.logger.info(f"[{cat}] Task [{i+1}/{len(cat_tasks)}]: {task.task_id} | {pct:.1f}% ({done}/{total}){eta_str}")
                    r = self.evaluate_single_task(task)
                    cat_results.append(r)
                    done += 1
                    # 使用"历史已完成 + 当前进度"决定保存频率和文件编号
                    # global_cat_progress 会在断点续评时自动递增，避免覆盖
                    global_cat_progress = completed_before + i + 1
                    if save_intermediate and ((global_cat_progress % 10 == 0) or (i+1) == len(cat_tasks)):
                        inc = cat_results[last_saved:]
                        if inc:
                            self._save_intermediate_results(inc, f"{cat}_{global_cat_progress}")
                            last_saved = len(cat_results)
                except Exception as e:
                    self.logger.error(f"[{cat}] Task {task.task_id} failed: {e}")
                    err_r = AudioCountingResult(
                        task_id=task.task_id,
                        category=task.category,
                        audio_path=task.audio_path,
                        predicted_count=None,
                        ground_truth=task.ground_truth,
                        is_correct=None,
                        processing_time=0.0,
                        error_message=str(e),
                        response=None,
                        annotation_index=(task.metadata or {}).get('annotation_index'),
                        question=task.prompt,
                        question_id=(task.metadata or {}).get('question_id')
                    )
                    cat_results.append(err_r)
                    done += 1
            if cat_results and save_after_each:
                self.save_results(cat_results)
            all_results.extend(cat_results)
        total_time = time.time() - start
        self.logger.info(f"All done: {total} tasks, {int(total_time//60)}m{int(total_time%60)}s")
        # 返回前做一次去重，避免报告重复统计
        return self._dedup_results(all_results)

    def _save_intermediate_results(self, results: List[AudioCountingResult], filename: str):
        """
        保存中间结果增量（仅写入本批新增的结果）。
        在当前运行目录下为每个类别创建子文件夹，并将本批增量写入
        "{category}_batch_progress_{filename}.json"。
        文件名中的数字基于全局进度（包含断点前已完成的任务数），因此断点续评时会自动递增。
        """
        by_cat: Dict[str, List[AudioCountingResult]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)
        root = self.results_dir
        root.mkdir(exist_ok=True, parents=True)
        
        for cat, cat_results in by_cat.items():
            cat_dir = root / f"{cat}_results"
            cat_dir.mkdir(exist_ok=True)
            
            # 直接使用传入的 filename（已包含类别和全局进度）
            fp = cat_dir / f"{cat}_batch_progress_{filename}.json"
            rows = []
            for r in cat_results:
                rows.append({
                    "task_id": r.task_id,
                    "audio_path": r.audio_path,
                    "predicted_count": r.predicted_count,
                    "ground_truth": r.ground_truth,
                    "is_correct": r.is_correct,
                    "processing_time": r.processing_time,
                    "absolute_error": r.absolute_error,
                    "squared_error": r.squared_error,
                    "memory_usage_mb": r.memory_usage_mb,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "raw_response": r.raw_response,
                    "response": r.response,
                    "annotation_index": r.annotation_index,
                    "question": r.question,
                    "question_id": r.question_id
                })
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

    def save_results(self, results: List[AudioCountingResult], filename: str = None):
        by_cat: Dict[str, List[AudioCountingResult]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cat, cat_results in by_cat.items():
            cat_dir = self.results_dir / f"{cat}_results"
            cat_dir.mkdir(exist_ok=True)
            if filename is None:
                fn = f"{cat}_results_{ts}.json"
            else:
                fn = f"{cat}_{filename}"
            fp = cat_dir / fn
            rows = []
            for r in cat_results:
                rows.append({
                    "task_id": r.task_id,
                    "audio_path": r.audio_path,
                    "predicted_count": r.predicted_count,
                    "ground_truth": r.ground_truth,
                    "is_correct": r.is_correct,
                    "processing_time": r.processing_time,
                    "absolute_error": r.absolute_error,
                    "squared_error": r.squared_error,
                    "memory_usage_mb": r.memory_usage_mb,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completed_tokens if hasattr(r, 'completed_tokens') else r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "raw_response": r.raw_response,
                    "response": r.response,
                    "annotation_index": r.annotation_index,
                    "question": r.question,
                    "question_id": r.question_id
                })
            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

    def generate_report(self, results: List[AudioCountingResult]) -> Dict[str, Any]:
        # 报告前安全去重，确保汇总统计覆盖“续测前+本次续测”且不重复
        results = self._dedup_results(results or [])
        if not results:
            return {"error": "no results"}
        total = len(results)
        succ = sum(1 for r in results if r.predicted_count is not None)
        tokens = sum((r.total_tokens or 0) for r in results)
        with_gt = [r for r in results if r.ground_truth is not None]
        correct = sum(1 for r in with_gt if r.is_correct is True)
        acc = correct/len(with_gt) if with_gt else 0
        avg_t = sum(r.processing_time for r in results)/total if total else 0
        by_cat: Dict[str, Dict[str, Any]] = {}
        for r in results:
            d = by_cat.setdefault(r.category, {
                'total_questions': 0,
                'total_audios': set(),
                'successful': 0,
                'correct': 0,
                'with_gt': 0
            })
            d['total_questions'] += 1
            d['total_audios'].add(r.audio_path)
            if r.predicted_count is not None:
                d['successful'] += 1
            if r.ground_truth is not None:
                d['with_gt'] += 1
                if r.is_correct is True:
                    d['correct'] += 1
        for cat in list(by_cat.keys()):
            d = by_cat[cat]
            d['total_audios'] = len(d['total_audios'])
            d['total'] = d['total_questions']
            d['accuracy'] = d['correct']/d['with_gt'] if d['with_gt'] else 0
            d['extraction_rate'] = d['successful']/d['total_questions'] if d['total_questions'] else 0
        return {
            'summary': {
                'total_tasks': total,
                'successful_extractions': succ,
                'extraction_rate': (succ/total) if total else 0,
                'tasks_with_ground_truth': len(with_gt),
                'correct_predictions': correct,
                'accuracy': acc,
                'average_processing_time': avg_t,
                'total_tokens': tokens
            },
            'by_category': by_cat,
            'timestamp': datetime.now().isoformat()
        }
