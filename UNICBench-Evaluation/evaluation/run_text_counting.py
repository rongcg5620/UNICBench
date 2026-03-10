#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本计数评测运行脚本
基于GPT-4o API进行文本计数评测
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluators.text_counting_evaluator import TextCountingEvaluator

try:
    from models.models_config import get_model_config, select_model_interactively, EVALUATION_SETTINGS as MULTI_MODEL_SETTINGS
    HAS_MULTI_MODEL_CONFIG = True
except ImportError:
    HAS_MULTI_MODEL_CONFIG = False
    # 兼容性回退
    try:
        from gpt4o_config import GPT4O_API_CONFIG, EVALUATION_SETTINGS
        HAS_LEGACY_CONFIG = True
    except ImportError:
        HAS_LEGACY_CONFIG = False


def create_timestamped_results_dir(selected_categories, text_dir_path: str, model_name: str) -> Path:
    """在 evaluation/results/text_results/{model_name}/ 下创建时间戳运行目录"""
    base = Path(__file__).parent / 'results' / 'text_results' / model_name
    base.mkdir(exist_ok=True, parents=True)

    try:
        total_available = len([d for d in Path(text_dir_path).iterdir() if d.is_dir()])
    except Exception:
        total_available = None

    label = 'all' if (total_available and len(selected_categories) == total_available) else f"{len(selected_categories)}cats"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base / f"text_{timestamp}_{label}"
    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir


def check_api_config(apis=None):
    """检查API配置是否正确"""
    if apis is None:
        apis = []
    
    if not apis:
        print("    Error: No API configuration available")
        return False
    
    # 检查 API key（DIRECT 类型不需要 key）
    if apis[0].get("type") != "DIRECT":
        api_key = apis[0].get("key", "")
        if "your-" in api_key and "-api-key" in api_key:
            print(f"    Error: Please configure your API key")
            print(f"    Replace '{api_key}' with your actual API key")
            return False
    return True


def check_data_directory():
    """检查数据目录是否存在"""
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "../UNICBench/text",       # 与evaluation同级的UNICBench目录
        script_dir / "../new_data/text",        # 原有的new_data目录
        script_dir / "../text",                 # 原有的text目录
        Path("../UNICBench/text"),              # 相对路径的UNICBench目录
        Path("../new_data/text"),               # 原有的相对路径
        Path("../text"),                        # 原有的相对路径
        Path("UNICBench/text"),                 # 当前目录下的UNICBench
        Path("new_data/text"),                  # 原有的当前目录
        Path("text")                            # 原有的当前目录
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return True, str(path)
    
    print("    Error: Data directory not found")
    print("    Please ensure there is a 'UNICBench/text', 'new_data/text' or 'text' folder with annotation files")
    return False, None
    print("    Error: Data directory not found")
    print("    Please ensure there is a 'new_data/text' or 'text' folder with annotation files")
    return False, None


def get_available_categories(data_dir_path):
    """从data_loader获取可用的类别"""
    from utils.data_loader import get_all_complate_data
    
    # 加载数据获取类别信息
    all_data = get_all_complate_data(data_dir_path, data_types=["text"])
    
    if not all_data:
        print("   Error: No tasks found in the data directory")
        return []
    
    # 从任务数据中提取类别
    categories = {}
    for task in all_data:
        file_path = Path(task['file_path'])
        # 从路径推断类别
        category = file_path.parent.name
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    return categories

def select_categories(data_dir_path):
    """选择要评测的类别"""
    categories = get_available_categories(data_dir_path)
    
    if not categories:
        return []
    
    print("Discovered categories:")
    category_list = list(categories.keys())
    for i, category in enumerate(category_list, 1):
        task_count = categories[category]
        print(f"   {i}. {category} ({task_count} tasks)")
    
    print("\nSelect categories to evaluate:")
    print("   a. All categories")
    print("   or enter category numbers, separated by commas (e.g.: 1,3,5)")
    
    choice = input("\nPlease select: ").strip()
    
    if choice.lower() == 'a':
        return category_list
    
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected = [category_list[i] for i in indices if 0 <= i < len(category_list)]
        return selected
    except (ValueError, IndexError):
        print("   Invalid selection, will evaluate all categories")
        return category_list


# ===== Helpers for resume-from-interruption (Text) =====
def _canonical_doc_key(p: Optional[str]) -> Optional[str]:
    """Normalize document path into a canonical key for dedup/lookup.
    Rules:
      - None -> None
      - lowercase
      - unify path separator to '/'
      - strip drive prefix like 'c:'
      - flatten 'evaluation/../' segment
      - if contains 'text/', keep tail from that anchor
      - else fall back to last 2-3 segments
    """
    if not p:
        return None
    s = p.replace('\\', '/').lower()
    if len(s) > 1 and s[1] == ':':
        s = s[2:]
    if s.startswith('/'):
        s = s[1:]
    s = s.replace('evaluation/../', '')
    anchor = 'text/'
    if anchor in s:
        s = s.split(anchor, 1)[1]
        return s
    parts = s.split('/')
    if len(parts) >= 3:
        return '/'.join(parts[-3:])
    return s


def _text_task_category(task) -> str:
    try:
        return Path(getattr(task, 'document_id', '')).parent.name or 'unknown'
    except Exception:
        return 'unknown'


def _record_key_text(rec: Dict[str, object]) -> Tuple:
    """Build a unique key tuple from a saved text result record."""
    task_obj = rec.get('task') if isinstance(rec, dict) else None
    doc = None
    qid = None
    if isinstance(task_obj, dict):
        doc = task_obj.get('document_id')
        qid = task_obj.get('question_id')
    else:
        doc = rec.get('document_id')
        qid = rec.get('question_id')
    dockey = _canonical_doc_key(doc)
    return ('doc_qid', dockey, qid)


def _build_text_processed_and_failed_keys(run_dir: Path, categories: List[str]) -> Tuple[Set[Tuple], Set[Tuple], Dict[str, int]]:
    """Scan existing text results under run_dir and compute processed/failed key sets.
    processed = predicted_count != -1
    failed    = predicted_count == -1
    Count processed_by_category using document folder name.
    """
    processed: Set[Tuple] = set()
    failed: Set[Tuple] = set()
    processed_by_category: Dict[str, int] = {}
    counted_by_category: Dict[str, Set[Tuple]] = {}

    if not run_dir.exists() or not run_dir.is_dir():
        return processed, failed, processed_by_category

    # 扫描所有的文本结果文件
    result_files = list(run_dir.glob('text_*.json'))
    result_files.extend(run_dir.glob('text_evaluation_*.json'))
    
    # 扫描各类别子目录的结果文件
    for subdir in run_dir.glob('*_results'):
        if subdir.is_dir():
            # 扫描子目录中的所有结果文件（包括 batch_progress 文件）
            for f in subdir.glob('*.json'):
                result_files.append(f)
    
    for result_file in result_files:
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            for record in data:
                try:
                    # 获取任务信息
                    task_info = record.get('task', {}) if isinstance(record, dict) else {}
                    doc_id = task_info.get('document_id')
                    question_id = task_info.get('question_id')
                    predicted_count = record.get('predicted_count')
                    
                    # 构造独特键
                    key = _record_key_text(record)
                    
                    # 判断是否成功/失败
                    if predicted_count is not None and predicted_count != -1:
                        processed.add(key)
                        # 统计每个类别已完成数量（使用文档目录名作为类别）
                        category = Path(doc_id).parent.name if doc_id else 'unknown'
                        if category in categories:
                            if category not in counted_by_category:
                                counted_by_category[category] = set()
                            if key not in counted_by_category[category]:
                                counted_by_category[category].add(key)
                                processed_by_category[category] = processed_by_category.get(category, 0) + 1
                    else:
                        failed.add(key)
                except Exception:
                    continue
        except Exception:
            continue
            
    return processed, failed, processed_by_category


def _generate_text_report_from_results(results: List[Dict]) -> Dict:
    """Generate a simple summary report from a list of result dicts (single session)."""
    total = len(results)
    valid = [r for r in results if r.get('predicted_count') is not None and r.get('predicted_count') != -1]
    succ = len(valid)

    # ground truth presence
    def _gt_of(r):
        t = r.get('task', {}) if isinstance(r, dict) else {}
        return t.get('ground_truth')

    with_gt = [r for r in results if _gt_of(r) is not None]
    correct = 0
    for r in with_gt:
        try:
            pred = r.get('predicted_count')
            gt = r.get('task', {}).get('ground_truth')
            if pred is not None and pred != -1 and gt is not None and pred == gt:
                correct += 1
        except Exception:
            pass

    # average processing time
    times = []
    for r in results:
        pt = r.get('processing_time')
        if pt is not None:
            try:
                times.append(float(pt))
            except Exception:
                pass
    avg_pt = (sum(times) / len(times)) if times else 0.0

    # token statistics (新增)
    total_tokens_used = 0
    prompt_tokens_used = 0
    completion_tokens_used = 0
    for r in results:
        try:
            if r.get('total_tokens'):
                total_tokens_used += int(r.get('total_tokens', 0))
            if r.get('prompt_tokens'):
                prompt_tokens_used += int(r.get('prompt_tokens', 0))
            if r.get('completion_tokens'):
                completion_tokens_used += int(r.get('completion_tokens', 0))
        except Exception:
            pass

    # by-category using document_id parent folder
    by_cat: Dict[str, Dict[str, float]] = {}
    for r in results:
        try:
            t = r.get('task', {})
            doc = t.get('document_id')
            cat = Path(doc).parent.name if doc else 'unknown'
            pred = r.get('predicted_count')
            gt = t.get('ground_truth')
            d = by_cat.setdefault(cat, {
                'total': 0, 'successful': 0, 'with_gt': 0, 'correct': 0
            })
            d['total'] += 1
            if pred is not None and pred != -1:
                d['successful'] += 1
            if gt is not None:
                d['with_gt'] += 1
                if pred is not None and pred != -1 and pred == gt:
                    d['correct'] += 1
        except Exception:
            continue
    # compute rates
    for cat, d in by_cat.items():
        d['extraction_rate'] = (d['successful'] / d['total']) if d['total'] else 0.0
        d['accuracy'] = (d['correct'] / d['with_gt']) if d['with_gt'] else 0.0

    return {
        'summary': {
            'total_tasks': total,
            'successful_extractions': succ,
            'extraction_rate': (succ / total) if total else 0.0,
            'tasks_with_ground_truth': len(with_gt),
            'correct_predictions': correct,
            'accuracy': (correct / len(with_gt)) if with_gt else 0.0,
            'average_processing_time': avg_pt,
            'total_tokens': total_tokens_used,
            'prompt_tokens': prompt_tokens_used,
            'completion_tokens': completion_tokens_used
        },
        'by_category': by_cat,
        'timestamp': datetime.now().isoformat(),
    }


def recompute_text_summary_report(run_dir: Path) -> Dict:
    """Recompute a combined report from all saved text result files under run_dir.
    Deduplicate by (canonical document path, question_id), keep latest occurrence.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Collect result files (支持新的按类别组织的结构)
    files: List[Path] = []
    
    # 1. 查找按类别组织的结果文件
    for category_dir in run_dir.iterdir():
        if not category_dir.is_dir() or not category_dir.name.endswith('_results'):
            continue
        
        category = category_dir.name.replace('_results', '')
        # 优先选择合并文件
        merged_file = category_dir / f"{category}_results_merged.json"
        if merged_file.exists():
            files.append(merged_file)
        else:
            # 检查是否有batch进度文件
            batch_files = sorted([f for f in category_dir.glob('*_batch_progress_*.json')])
            # 检查是否有最终结果文件
            final_files = sorted([f for f in category_dir.glob('*.json') if 'batch_progress' not in f.name])
            
            # 如果有多个batch文件，优先使用所有batch文件（更完整）
            if len(batch_files) > 1:
                # Resume模式：多个batch文件说明是分批完成的，需要全部读取
                files.extend(batch_files)
            elif final_files:
                # 只有最终文件，使用最新的最终结果
                files.append(final_files[-1])
            elif batch_files:
                # 只有一个batch文件，使用它
                files.extend(batch_files)
    
    # 2. 如果没有找到按类别组织的文件，回退为旧格式
    if not files:
        finals = sorted(run_dir.glob('text_evaluation_results_*.json'))
        inters = sorted(run_dir.glob('text_evaluation_intermediate_*.json'))
        files.extend(finals)
        if not finals:
            files.extend(inters)

    # Load and dedupe
    seen: Dict[Tuple, Dict] = {}
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            rows = data if isinstance(data, list) else [data]
            for r in rows:
                try:
                    t = r.get('task', {}) if isinstance(r, dict) else {}
                    doc = t.get('document_id')
                    qid = t.get('question_id')
                    key = ('doc_qid', _canonical_doc_key(doc), qid)
                    seen[key] = r  # later overwrite earlier
                except Exception:
                    continue
        except Exception:
            continue

    combined = list(seen.values())
    report = _generate_text_report_from_results(combined)
    with open(run_dir / 'text_counting_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report

def main():
    """主函数"""
    print("   Multi-Model Text Counting Evaluation System")
    print("=" * 50)
    
    # 1. 选择模型
    if HAS_MULTI_MODEL_CONFIG:
        print("\n   Model Selection")
        print("-" * 30)
        selected_model_name = select_model_interactively()  # 使用交互式选择
        if not selected_model_name:
            print("   No model selected, exiting...")
            return
        
        try:
            selected_apis = get_model_config(selected_model_name)
            evaluation_settings = MULTI_MODEL_SETTINGS
            print(f"   Using model: {selected_model_name}")
            # API配置加载成功
        except Exception as e:
            print(f"   Error loading model config: {e}")
            print("   Error: No model selected")
            return
    elif HAS_LEGACY_CONFIG:
        print("\n   Using legacy GPT-4o configuration")
        selected_apis = GPT4O_API_CONFIG
        evaluation_settings = EVALUATION_SETTINGS
        selected_model_name = "gpt-4o"
    else:
        print("   Error: No model configuration found")
        print("   Please ensure models_config.py or gpt4o_config.py exists")
        return
    
    # 2. 检查API配置
    if not check_api_config(selected_apis):
        return
    
    # 2. 检查数据目录
    data_check_result = check_data_directory()
    if isinstance(data_check_result, tuple):
        data_ok, data_dir_path = data_check_result
        if not data_ok:
            return
    else:
        return

    # 3. 选择评测类别
    selected_categories = select_categories(data_dir_path)
    if not selected_categories:
        print("   No categories selected")
        return
    
    print(f"\n   Will evaluate the following categories: {', '.join(selected_categories)}")
    
    # 4. 设置任务数量限制
    max_tasks = input(f"\nMaximum number of texts to evaluate per category? (default: {evaluation_settings['max_tasks_per_category']}): ").strip()
    try:
        max_tasks = int(max_tasks) if max_tasks else evaluation_settings['max_tasks_per_category']
    except ValueError:
        max_tasks = evaluation_settings['max_tasks_per_category']
    
    print(f"  Maximum of {max_tasks} texts per category will be evaluated")

    # 5. 设置批量模式选项
    batch_mode_input = input(f"\nEnable batch mode for multiple questions per document? (Y/n): ").strip().lower()
    batch_mode = batch_mode_input != 'n'
    
    if batch_mode:
        print(f"    Batch mode enabled: Multiple questions per document will be processed together")
    else:
        print(f"    Batch mode disabled: Each question will be processed individually")
    
    # 6. 设置可视化选项
    auto_viz = input(f"\nGenerate visualization analysis after evaluation? (Y/n): ").strip().lower()
    enable_visualization = auto_viz != 'n'

    if enable_visualization:
        print(f"    Visualization will be generated automatically")
    else:
        print(f"    Visualization skipped (can run manually later)")

    # 6. 选择是创建新运行目录还是从已存在目录进行断点续评
    resume_choice = input("\nResume from an existing results directory? (y/N): ").strip().lower()
    resume_mode = (resume_choice == 'y')
    run_results_dir: Path
    model_name: str
    if resume_mode:
        resume_path = input("Enter existing run dir (e.g., evaluation/results/text_results/<model>/text_YYYYMMDD_HHMMSS_*): ").strip()
        rp = Path(resume_path)
        if rp.exists() and rp.is_dir():
            run_results_dir = rp
            # 尝试从路径推断模型名
            try:
                # 新的路径结构: results/text_results/{model_name}/text_{timestamp}_...
                model_name = run_results_dir.parent.name
            except Exception:
                model_name = selected_model_name
            print(f"\n   Resuming evaluation under: {run_results_dir}")
        else:
            print("   Invalid path, creating a new run directory instead.")
            model_name = input(f"\nEnter model name for results folder (default: {selected_model_name}): ").strip() or selected_model_name
            run_results_dir = create_timestamped_results_dir(selected_categories, data_dir_path, model_name)
            print(f"\n   Results will be saved under: {run_results_dir}")
            resume_mode = False
    else:
        # 创建新运行目录 evaluation/results/{model_name}/text_{timestamp}_<label>
        model_name = input(f"\nEnter model name for results folder (default: {selected_model_name}): ").strip() or selected_model_name
        run_results_dir = create_timestamped_results_dir(selected_categories, data_dir_path, model_name)
        print(f"\n   Results will be saved under: {run_results_dir}")

    # 7. 创建评测器
    print("\n   Initializing evaluator...")
    evaluator = TextCountingEvaluator(
        apis=selected_apis,
        max_try=evaluation_settings.get("max_try", 6),
        save_dir=str(run_results_dir),
        batch_mode=batch_mode
    )

    # 8. 生成任务
    print("   Loading text tasks...")
    all_tasks = evaluator.load_text_tasks(data_dir_path, categories=selected_categories)
    
    # 简化的任务过滤逻辑
    filtered_tasks = []
    category_counts = {}
    
    for task in all_tasks:
        # 从文档ID推断类别
        category = Path(task.document_id).parent.name if hasattr(task, 'document_id') else 'unknown'
        
        # 检查任务类别是否在选中的类别中
        if category in selected_categories:
            if category not in category_counts:
                category_counts[category] = 0
            
            # 简单的数量限制
            if category_counts[category] < max_tasks:
                filtered_tasks.append(task)
                category_counts[category] += 1

    # 8.5 断点续评：在恢复模式下跳过已完成任务，可选择是否重试失败任务，并按配额补齐
    if resume_mode:
        retry_failed_choice = input("\nRetry previously failed tasks recorded in this run? (Y/n): ").strip().lower()
        retry_failed = (retry_failed_choice != 'n')
        processed_keys, failed_keys, processed_by_category = _build_text_processed_and_failed_keys(run_results_dir, selected_categories)

        # 先在当前 filtered_tasks 中剔除已完成/需要跳过失败的
        remaining_tasks = []
        existing_keys: Set[Tuple] = set()
        for t in filtered_tasks:
            dockey = _canonical_doc_key(getattr(t, 'document_id', None))
            key = ('doc_qid', dockey, getattr(t, 'question_id', None))
            if key in processed_keys:
                continue
            if (not retry_failed) and (key in failed_keys):
                continue
            remaining_tasks.append(t)
            existing_keys.add(key)

        # 计算各类别剩余额度：最大数 - 历史已处理数
        needed_by_cat: Dict[str, int] = {}
        for cat in selected_categories:
            already = processed_by_category.get(cat, 0)
            needed_by_cat[cat] = max(0, max_tasks - already)

        # 当前剩余任务按类计数
        current_counts: Dict[str, int] = {}
        for t in remaining_tasks:
            cat = _text_task_category(t)
            current_counts[cat] = current_counts.get(cat, 0) + 1

        # 从 all_tasks 继续补充达到各类别额度
        for t in all_tasks:
            cat = _text_task_category(t)
            if cat not in selected_categories:
                continue
            # 若该类别额度已满则跳过
            if current_counts.get(cat, 0) >= needed_by_cat.get(cat, 0):
                continue
            dockey = _canonical_doc_key(getattr(t, 'document_id', None))
            key = ('doc_qid', dockey, getattr(t, 'question_id', None))
            if key in existing_keys:
                continue
            if key in processed_keys:
                continue
            if (not retry_failed) and (key in failed_keys):
                continue
            remaining_tasks.append(t)
            existing_keys.add(key)
            current_counts[cat] = current_counts.get(cat, 0) + 1
            # 可选：若所有类别均达到额度，可提前结束
            all_met = True
            for c in selected_categories:
                if current_counts.get(c, 0) < needed_by_cat.get(c, 0):
                    all_met = False
                    break
            if all_met:
                break

        # 若各类别额度为0且 remaining_tasks 为空，说明无需继续
        filtered_tasks = remaining_tasks
        if not filtered_tasks:
            print("   Nothing to do: budgets are already met by previous results (and no failed retry requested).")
            # 在续评但无任务情况下，仍然重算报告
            try:
                print("   Recomputing summary report...")
                _ = recompute_text_summary_report(run_results_dir)
            except Exception:
                pass
            return
    
    if not filtered_tasks:
        print("   No matching tasks found")
        return
    
    print(f"   Generated {len(filtered_tasks)} evaluation tasks")
    
    # 按类别显示任务分布（若为续评模式，按续评后的任务重算分布）
    print("\n    Task distribution:")
    if resume_mode:
        category_counts_remain: Dict[str, int] = {}
        for t in filtered_tasks:
            cat = _text_task_category(t)
            category_counts_remain[cat] = category_counts_remain.get(cat, 0) + 1
        for category, count in category_counts_remain.items():
            print(f"   {category}: {count} tasks")
    else:
        for category, count in category_counts.items():
            print(f"   {category}: {count} tasks")
    
    # 9. 确认开始评测
    confirm = input(f"\nStart evaluation? (y/N): ").strip().lower()
    if confirm != 'y':
        print("   Evaluation cancelled")
        return

    # 10. 执行评测（按类别评测 + 开启增量中间保存）
    print("\n   Starting evaluation (by category with intermediate saves)...")
    try:
        results = evaluator.evaluate_by_category(
            filtered_tasks,
            save_after_each=True,
            save_intermediate=True
        )
        
        # 11. 在断点续评模式下，重建本轮涉及类别的最终结果文件（合并历史+增量）
        if resume_mode:
            cats_touched = sorted({_text_task_category(t) for t in filtered_tasks})
            if cats_touched:
                print("\n   Rebuilding category final results (merged) for:", ", ".join(cats_touched))
                # TODO: 实现文本类别的结果重建
                pass

        # 12. 生成并保存报告
        if resume_mode:
            # 在断点续评模式下，基于已保存的详细结果（包含历史+新增）复算并写回最终报告
            try:
                combined_report = recompute_text_summary_report(run_results_dir)
                summary = combined_report['summary']
                report_path = run_results_dir / "text_counting_report.json"
                print("\n" + "=" * 50)
                print("   Text Evaluation Report (Resumed Run)")
                print("=" * 50)
                print(f"Total tasks: {summary['total_tasks']}")
                print(f"Successful extractions: {summary['successful_extractions']}")
                print(f"Extraction rate: {summary['extraction_rate']:.2%}")
                if summary['tasks_with_ground_truth'] > 0:
                    print(f"Tasks with ground truth: {summary['tasks_with_ground_truth']}")
                    print(f"Accuracy: {summary['accuracy']:.2%}")
                else:
                    print("Note: No ground truth available for accuracy comparison")
                print(f"Average processing time: {summary['average_processing_time']:.2f} seconds")
                
                # 显示token统计信息
                if summary.get('total_tokens', 0) > 0:
                    print(f"Total tokens used: {summary['total_tokens']:,}")
                    print(f"  - Prompt tokens: {summary.get('prompt_tokens', 0):,}")
                    print(f"  - Completion tokens: {summary.get('completion_tokens', 0):,}")
                
                # 显示批量模式统计（如果可用）
                if summary.get('batch_mode_enabled'):
                    print(f"\nBatch Mode Statistics:")
                    print(f"  - Batch API calls: {summary.get('batch_api_calls', 0)}")
                    print(f"  - Single API calls: {summary.get('single_api_calls', 0)}")
                    print(f"  - Total API calls: {summary.get('total_api_calls', 0)}")
                
                print(f"   Report saved to: {report_path}")
            except Exception as e:
                # 回退：仅基于本轮结果生成报告（不含历史）
                print(f"   Warning: Failed to recompute combined report: {e}")
                results_dict = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
                report = _generate_text_report_from_results(results_dict)
                with open(run_results_dir / 'text_counting_report.json', 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                summary = report['summary']
                print("\n" + "=" * 50)
                print("   Text Evaluation Report (This Session)")
                print("=" * 50)
                print(f"Total tasks: {summary['total_tasks']}")
                print(f"Successful extractions: {summary['successful_extractions']}")
                print(f"Extraction rate: {summary['extraction_rate']:.2%}")
                if summary['tasks_with_ground_truth'] > 0:
                    print(f"Tasks with ground truth: {summary['tasks_with_ground_truth']}")
                    print(f"Accuracy: {summary['accuracy']:.2%}")
                else:
                    print("Note: No ground truth available for accuracy comparison")
                print(f"Average processing time: {summary['average_processing_time']:.2f} seconds")
                
                # 显示token统计信息
                if summary.get('total_tokens', 0) > 0:
                    print(f"Total tokens used: {summary['total_tokens']:,}")
                    print(f"  - Prompt tokens: {summary.get('prompt_tokens', 0):,}")
                    print(f"  - Completion tokens: {summary.get('completion_tokens', 0):,}")
        else:
            # 新运行：基于本轮结果生成完整报告
            results_dict = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
            report = _generate_text_report_from_results(results_dict)
            with open(run_results_dir / 'text_counting_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print("\n" + "=" * 50)
            print("   Text Evaluation Report")
            print("=" * 50)
            
            summary = report['summary']
            print(f"Total tasks: {summary['total_tasks']}")
            print(f"Successful extractions: {summary['successful_extractions']}")
            print(f"Extraction rate: {summary['extraction_rate']:.2%}")
            
            if summary['tasks_with_ground_truth'] > 0:
                print(f"Tasks with ground truth: {summary['tasks_with_ground_truth']}")
                print(f"Accuracy: {summary['accuracy']:.2%}")
            else:
                print("Note: No ground truth available for accuracy comparison")
            
            print(f"Average processing time: {summary['average_processing_time']:.2f} seconds")
            
            # 显示token统计信息
            if summary.get('total_tokens', 0) > 0:
                print(f"Total tokens used: {summary['total_tokens']:,}")
                print(f"  - Prompt tokens: {summary.get('prompt_tokens', 0):,}")
                print(f"  - Completion tokens: {summary.get('completion_tokens', 0):,}")
            
            # 显示批量模式统计（如果可用）
            if summary.get('batch_mode_enabled'):
                print(f"\nBatch Mode Statistics:")
                print(f"  - Batch API calls: {summary.get('batch_api_calls', 0)}")
                print(f"  - Single API calls: {summary.get('single_api_calls', 0)}")
                print(f"  - Total API calls: {summary.get('total_api_calls', 0)}")
        
        print(f"\n   Evaluation completed!")

        # 可选：同时打印一次本轮摘要
        evaluator._print_summary(results)
        
        # 13. 自动生成可视化分析（如果用户选择了）
        if enable_visualization:
            print(f"\n   Generating visualization analysis...")
            try:
                from text_visualization_analyzer import TextCountingVisualizationAnalyzer, TextVisualizationConfig

                # 创建可视化配置
                viz_config = TextVisualizationConfig(
                    figure_size=(12, 8),
                    dpi=200,  # 中等分辨率，平衡质量和速度
                    style='default',  # 使用默认样式确保兼容性
                    color_palette='Set2'
                )

                # 创建可视化分析器
                analyzer = TextCountingVisualizationAnalyzer(viz_config)

                # 加载刚刚生成的结果（从本次运行目录）
                viz_results = analyzer.load_results(str(run_results_dir))

                if viz_results:
                    # 生成可视化输出目录（位于本次运行目录下）
                    viz_output_dir = str(run_results_dir / 'visualization')

                    # 生成完整的可视化仪表板
                    analyzer.create_comprehensive_dashboard(viz_results, viz_output_dir)

                    # 获取关键指标用于显示
                    metrics = analyzer.calculate_comprehensive_metrics(viz_results)
                    overall = metrics['overall']

                    print(f"    Text visualization analysis completed!")
                    print(f"    Key metrics:")
                    print(f"      - Overall accuracy: {overall['accuracy']:.3f}")
                    print(f"      - Mean Absolute Error: {overall['mae']:.3f}")
                    print(f"      - Acc@±1: {overall.get('acc_at_1', 0):.3f}")
                    print(f"      - Acc@±2: {overall.get('acc_at_2', 0):.3f}")
                    print(f"    Visualization dashboard saved to: {viz_output_dir}")
                    print(f"    View comprehensive_report.md for detailed analysis")

                else:
                    print(f"    Could not load results for visualization")

            except ImportError:
                print(f"    Visualization dependencies not available")
                print(f"    Install with: pip install matplotlib seaborn numpy pandas")
            except Exception as viz_error:
                print(f"    Visualization generation failed: {str(viz_error)}")
                print(f"    You can manually run: python text_visualization_analyzer.py {run_results_dir}")
    except KeyboardInterrupt:
        print("\n   Evaluation interrupted by user")
    except Exception as e:
        print(f"\n   Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
