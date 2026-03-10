#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片计数评测运行脚本
基于GPT-4o API进行图片计数评测
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

from evaluators.image_counting_evaluator import ImageCountingEvaluator

try:
    from models.models_config import get_model_config, select_model_interactively, EVALUATION_SETTINGS as MULTI_MODEL_SETTINGS
    HAS_MULTI_MODEL_CONFIG = True
except ImportError:
    HAS_MULTI_MODEL_CONFIG = False


def create_timestamped_results_dir(selected_categories, images_dir_path: str, model_name: str) -> Path:
    """在 evaluation/results/image_results/{model_name}/ 下创建时间戳运行目录"""
    base = Path(__file__).parent / 'results' / 'image_results' / model_name
    base.mkdir(exist_ok=True, parents=True)

    try:
        total_available = len([d for d in Path(images_dir_path).iterdir() if d.is_dir()])
    except Exception:
        total_available = None

    label = 'all' if (total_available and len(selected_categories) == total_available) else f"{len(selected_categories)}cats"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base / f"{timestamp}_{label}"
    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir


def check_api_config(apis=None):
    """检查API配置是否正确"""
    if apis is None:
        apis = []
    
    api_key = apis[0]["key"]
    if "your-" in api_key and "-api-key" in api_key:
        print(f"    Error: Please configure your API key")
        print(f"    Replace '{api_key}' with your actual API key")
        return False
    return True


def check_data_directory():
    """检查数据目录是否存在"""
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir / "../UNICBench/image",      # 与evaluation同级的UNICBench目录
        script_dir / "../new_data/image",       # 原有的new_data目录
        Path("../UNICBench/image"),             # 相对路径的UNICBench目录
        Path("../new_data/image"),              # 原有的相对路径
        Path("UNICBench/image"),                # 当前目录下的UNICBench
        Path("new_data/image")                  # 原有的当前目录
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            return True, str(path)
    
    print("    Error: Data directory not found")
    print("    Please ensure there is a 'UNICBench/image' or 'new_data/image' folder with annotation files")
    return False, None


def get_available_categories(data_dir_path):
    """从data_loader获取可用的类别"""
    from utils.data_loader import get_all_complate_data
    
    # 加载数据获取类别信息
    all_data = get_all_complate_data(data_dir_path, data_types=["image"])
    
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


# ===== Helpers for resume-from-interruption =====
def _canonical_image_key(p: Optional[str]) -> Optional[str]:
    """Normalize image path into a canonical key for dedup/lookup.
    Rules:
      - None -> None
      - lowercase
      - unify path separator to '/'
      - strip drive prefix like 'c:'
      - flatten 'evaluation/../' segment
      - if contains 'new_data/image/', keep tail from that anchor
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
    anchor = 'new_data/image/'
    if anchor in s:
        s = s.split(anchor, 1)[1]
        return s
    parts = s.split('/')
    if len(parts) >= 3:
        return '/'.join(parts[-3:])
    return s


def _record_key(rec: Dict[str, object]) -> Tuple:
    """Build a unique key tuple from a saved result record."""
    imgk = _canonical_image_key(rec.get('image_path'))
    qid = rec.get('question_id')
    ann = rec.get('annotation_index')
    if imgk is not None and qid is not None:
        return ('img_qid', imgk, qid)
    if imgk is not None and ann is not None:
        return ('img_ann', imgk, ann)
    task_id = rec.get('task_id')
    return ('task', task_id)


def _task_key(task) -> Tuple:
    """Build a unique key tuple from an ImageCountingTask."""
    imgk = _canonical_image_key(getattr(task, 'image_path', None))
    meta: Dict[str, object] = getattr(task, 'metadata', None) or {}
    qid = meta.get('question_id')
    ann = meta.get('annotation_index')
    if imgk is not None and qid is not None:
        return ('img_qid', imgk, qid)
    if imgk is not None and ann is not None:
        return ('img_ann', imgk, ann)
    return ('task', getattr(task, 'task_id', None))


def _build_processed_and_failed_keys(run_dir: Path, categories: List[str]) -> Tuple[Set[Tuple], Set[Tuple], Dict[str, int]]:
    """Scan existing results under run_dir and compute processed/failed key sets.
    Processed = predicted_count is not None
    Failed    = predicted_count is None
    We read all *.json files under each category_results/ folder to be robust.
    """
    processed: Set[Tuple] = set()
    failed: Set[Tuple] = set()
    processed_by_category: Dict[str, int] = {}
    counted_by_category: Dict[str, Set[Tuple]] = {}
    for category in categories:
        cat_dir = run_dir / f"{category}_results"
        if not cat_dir.exists() or not cat_dir.is_dir():
            continue
        for jf in cat_dir.glob('*.json'):
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for rec in data:
                    try:
                        key = _record_key(rec)
                        pred = rec.get('predicted_count')
                        if pred is None:
                            failed.add(key)
                        else:
                            processed.add(key)
                            # 统计每个类别已完成数量（使用目录名作为类别），避免重复计数
                            if category not in counted_by_category:
                                counted_by_category[category] = set()
                            if key not in counted_by_category[category]:
                                counted_by_category[category].add(key)
                                processed_by_category[category] = processed_by_category.get(category, 0) + 1
                    except Exception:
                        continue
            except Exception:
                continue
    return processed, failed, processed_by_category


def _rebuild_category_final_results(run_dir: Path, category: str) -> Optional[Path]:
    """Scan all JSON files under <run_dir>/<category>_results and rebuild a merged
    final results file '<category>_results_merged.json' with de-duplication.
    De-dup keys:
      1) (canonical_image_path, question_id)
      2) (canonical_image_path, annotation_index)
      3) fallback: task_id
    """
    try:
        cat_dir = run_dir / f"{category}_results"
        if not cat_dir.exists() or not cat_dir.is_dir():
            return None
        # Load all records from all json files (including batch_progress and final ones)
        all_records: List[Dict[str, object]] = []
        for jf in cat_dir.glob('*.json'):
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict):
                    all_records.append(data)
            except Exception:
                continue

        if not all_records:
            return None

        # Deduplicate keeping the last occurrence
        dedup: Dict[Tuple, Dict[str, object]] = {}
        for r in all_records:
            try:
                img = r.get('image_path')
                imgk = _canonical_image_key(img)
                qid = r.get('question_id')
                ann = r.get('annotation_index')
                task_id = r.get('task_id')
                if imgk is not None and qid is not None:
                    key = ('img_qid', imgk, qid)
                elif imgk is not None and ann is not None:
                    key = ('img_ann', imgk, ann)
                elif task_id is not None:
                    key = ('task', task_id)
                else:
                    key = ('row_index', id(r))
                dedup[key] = r
            except Exception:
                continue

        merged = list(dedup.values())
        merged_path = cat_dir / f"{category}_results_merged.json"
        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"   Rebuilt final results for category '{category}': {merged_path}")
        return merged_path
    except Exception:
        return None


def _rebuild_final_results_for_categories(run_dir: Path, categories: List[str]) -> List[Path]:
    built: List[Path] = []
    for cat in categories:
        p = _rebuild_category_final_results(run_dir, cat)
        if p is not None:
            built.append(p)
    return built


def main():
    """主函数"""
    print("   Multi-Model Image Counting Evaluation System")
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
    else:
        print("   Error: No model selected")
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
    max_tasks = input(f"\nMaximum number of images to evaluate per category? (default: {evaluation_settings['max_tasks_per_category']}): ").strip()
    try:
        max_tasks = int(max_tasks) if max_tasks else evaluation_settings['max_tasks_per_category']
    except ValueError:
        max_tasks = evaluation_settings['max_tasks_per_category']
    
    print(f"  Maximum of {max_tasks} images per category will be evaluated")

    # 5. 设置可视化选项
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
        resume_path = input("Enter existing run dir (e.g., evaluation/results/image_results/<model>/<timestamp>_...): ").strip()
        rp = Path(resume_path)
        if rp.exists() and rp.is_dir():
            run_results_dir = rp
            # 尝试从路径推断模型名
            try:
                # 新的路径结构: results/image_results/{model_name}/{timestamp}_...
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
        # 创建新运行目录 evaluation/results/{model_name}/{timestamp}_<label>
        model_name = input(f"\nEnter model name for results folder (default: {selected_model_name}): ").strip() or selected_model_name
        run_results_dir = create_timestamped_results_dir(selected_categories, data_dir_path, model_name)
        print(f"\n   Results will be saved under: {run_results_dir}")

    # 7. 创建评测器
    print("\n   Initializing evaluator...")
    evaluator = ImageCountingEvaluator(
        apis=selected_apis,
        max_try=evaluation_settings["max_try"],
        results_dir=str(run_results_dir)
    )

    # 8. 生成任务
    print("   Generating evaluation tasks...")
    all_tasks = evaluator.generate_tasks_from_images_dir(data_dir_path)
    
    # 简化的任务过滤逻辑
    filtered_tasks = []
    category_counts = {}
    
    for task in all_tasks:
        # 检查任务类别是否在选中的类别中
        if task.category in selected_categories:
            if task.category not in category_counts:
                category_counts[task.category] = 0
            
            # 简单的数量限制
            if category_counts[task.category] < max_tasks:
                filtered_tasks.append(task)
                category_counts[task.category] += 1
    
    # 7.5 断点续评：在恢复模式下跳过已完成任务，可选择是否重试失败任务
    if resume_mode:
        retry_failed_choice = input("\nRetry previously failed tasks recorded in this run? (Y/n): ").strip().lower()
        retry_failed = (retry_failed_choice != 'n')
        processed_keys, failed_keys, processed_by_category = _build_processed_and_failed_keys(run_results_dir, selected_categories)

        # 先在当前 filtered_tasks 中剔除已完成/需要跳过失败的
        remaining_tasks = []
        existing_keys: Set[Tuple] = set()
        for t in filtered_tasks:
            k = _task_key(t)
            if k in processed_keys:
                continue
            if (not retry_failed) and (k in failed_keys):
                continue
            remaining_tasks.append(t)
            existing_keys.add(k)

        # 计算各类别剩余额度：最大数 - 已处理数
        needed_by_cat: Dict[str, int] = {}
        for cat in selected_categories:
            already = processed_by_category.get(cat, 0)
            needed_by_cat[cat] = max(0, max_tasks - already)

        # 统计当前剩余任务中各类别数量
        current_counts: Dict[str, int] = {}
        for t in remaining_tasks:
            current_counts[t.category] = current_counts.get(t.category, 0) + 1

        # 从 all_tasks 继续补充，直至达到各类别额度
        for t in all_tasks:
            if t.category not in selected_categories:
                continue
            # 若该类别额度已满则跳过
            if current_counts.get(t.category, 0) >= needed_by_cat.get(t.category, 0):
                continue
            k = _task_key(t)
            if k in existing_keys:
                continue
            if k in processed_keys:
                continue
            if (not retry_failed) and (k in failed_keys):
                continue
            remaining_tasks.append(t)
            existing_keys.add(k)
            current_counts[t.category] = current_counts.get(t.category, 0) + 1
            # 可选：若所有类别均达到额度，可提前结束
            all_met = True
            for cat in selected_categories:
                if current_counts.get(cat, 0) < needed_by_cat.get(cat, 0):
                    all_met = False
                    break
            if all_met:
                break

        # 若各类别额度为0且 remaining_tasks 为空，说明无需继续
        filtered_tasks = remaining_tasks
        if not filtered_tasks:
            print("   Nothing to do: budgets are already met by previous results (and no failed retry requested).")
            # 在续评但无任务情况下，仍然为所选类别重建合并后的最终结果文件，避免只剩 batch_progress 导致汇总/可视化不完整
            try:
                print("   Rebuilding category final results (merged) for:", ", ".join(selected_categories))
                _ = _rebuild_final_results_for_categories(run_results_dir, selected_categories)
                # 重算一次总报告
                from rerun_failed_images import recompute_summary_report
                _ = recompute_summary_report(run_results_dir)
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
            category_counts_remain[t.category] = category_counts_remain.get(t.category, 0) + 1
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
        # 中间结果文件名会自动基于图片编号（如 0060.jpg -> potatoes_batch_progress_potatoes_60.json）
        results = evaluator.evaluate_by_category(
            filtered_tasks,
            save_after_each=True,
            save_intermediate=True
        )

        # 11. 在断点续评模式下，重建本轮涉及类别的最终结果文件（合并历史+增量）
        if resume_mode:
            cats_touched = sorted({t.category for t in filtered_tasks})
            if cats_touched:
                print("\n   Rebuilding category final results (merged) for:", ", ".join(cats_touched))
                _ = _rebuild_final_results_for_categories(run_results_dir, cats_touched)

        # 12. 生成并保存报告
        if resume_mode:
            # 在断点续评模式下，基于已保存的详细结果（包含历史+新增）复算并写回最终报告
            try:
                from rerun_failed_images import recompute_summary_report
                combined_report = recompute_summary_report(run_results_dir)
                summary = combined_report['summary']
                report_path = run_results_dir / "image_counting_report.json"
                print("\n" + "=" * 50)
                print("   Evaluation Report (Resumed Run)")
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
                print(f"   Report saved to: {report_path}")
            except Exception as _:
                # 回退：仅基于本轮结果生成报告（不含历史）
                report = evaluator.generate_report(results)
                report_path = run_results_dir / "image_counting_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                summary = report['summary']
                print("\n" + "=" * 50)
                print("   Evaluation Report (This Session)")
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
                print(f"   Report saved to: {report_path}")
        else:
            # 新运行：基于本轮结果生成完整报告
            report = evaluator.generate_report(results)
            report_path = run_results_dir / "image_counting_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print("\n" + "=" * 50)
            print("   Evaluation Report")
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
        # print(f"Total tokens used: {summary['total_tokens']}")
        
        # 按类别显示结果
        # print(f"\n   Statistics by category:")
        # for category, stats in report['by_category'].items():
        #     print(f"  {category}:")
        #     print(f"    - Total tasks: {stats['total']}")
        #     print(f"    - Successful extractions: {stats['successful']} (Success rate: {stats['extraction_rate']:.2%})")
        #     if stats['with_gt'] > 0:
        #         print(f"    - Accuracy: {stats['accuracy']:.2%}")
        
        print(f"\n   Evaluation completed!")
        # print(f"   Detailed results saved to:")
        # for path in result_paths:
        #     print(f"   - {path}")
        # report_path 已在上面输出

        # 13. 自动生成可视化分析（如果用户选择了）
        if enable_visualization:
            print(f"\n   Generating visualization analysis...")
            try:
                from visualization_analyzer import CountingVisualizationAnalyzer, VisualizationConfig

                # 创建可视化配置
                viz_config = VisualizationConfig(
                    figure_size=(12, 8),
                    dpi=200,  # 中等分辨率，平衡质量和速度
                    style='default',  # 使用默认样式确保兼容性
                    color_palette='Set2'
                )

                # 创建可视化分析器
                analyzer = CountingVisualizationAnalyzer(viz_config)

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

                    print(f"    Visualization analysis completed!")
                    print(f"    Key metrics:")
                    print(f"      - Overall accuracy: {overall['accuracy']:.3f}")
                    print(f"      - Mean Absolute Error: {overall['mae']:.3f}")
                    print(f"      - Acc@±1: {overall['acc_at_1']:.3f}")
                    print(f"      - Acc@±2: {overall['acc_at_2']:.3f}")
                    print(f"    Visualization dashboard saved to: {viz_output_dir}")
                    print(f"    View comprehensive_report.md for detailed analysis")

                else:
                    print(f"    Could not load results for visualization")

            except ImportError:
                print(f"    Visualization dependencies not available")
                print(f"    Install with: pip install matplotlib seaborn numpy pandas psutil")
            except Exception as viz_error:
                print(f"    Visualization generation failed: {str(viz_error)}")
                print(f"    You can manually run: python run_visualization.py")

    except KeyboardInterrupt:
        print("\n   Evaluation interrupted by user")
    except Exception as e:
        print(f"\n   Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
