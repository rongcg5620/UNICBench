#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluators.audio_counting_evaluator import AudioCountingEvaluator

try:
    from models.models_config import get_model_config, select_model_interactively, EVALUATION_SETTINGS as MULTI_MODEL_SETTINGS
    HAS_MULTI_MODEL_CONFIG = True
except ImportError:
    HAS_MULTI_MODEL_CONFIG = False


def create_timestamped_results_dir(selected_categories: List[str], audio_dir_path: str, model_name: str) -> Path:
    base = Path(__file__).parent / 'results' / 'audio_results' / model_name
    base.mkdir(exist_ok=True, parents=True)
    try:
        total_available = len([d for d in Path(audio_dir_path).iterdir() if d.is_dir()])
    except Exception:
        total_available = None
    label = 'all' if (total_available and len(selected_categories) == total_available) else f"{len(selected_categories)}cats"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base / f"audio_{timestamp}_{label}"
    run_dir.mkdir(exist_ok=True, parents=True)
    return run_dir


def check_api_config(apis=None):
    if apis is None:
        apis = []
    if not apis:
        print("    Error: No API configuration available")
        return False
    if apis[0].get("type") != "DIRECT":
        api_key = apis[0].get("key", "")
        if "your-" in api_key and "-api-key" in api_key:
            print("    Error: Please configure your API key")
            return False
    return True


def check_data_directory():
    script_dir = Path(__file__).parent
    possible = [
        script_dir / "../UNICBench/audio",      # 与evaluation同级的UNICBench目录
        script_dir / "../new_data/audio",       # 原有的new_data目录
        Path("../UNICBench/audio"),             # 相对路径的UNICBench目录
        Path("../new_data/audio"),              # 原有的相对路径
        Path("UNICBench/audio"),                # 当前目录下的UNICBench
        Path("new_data/audio"),                 # 原有的当前目录
    ]
    for p in possible:
        if p.exists() and p.is_dir():
            return True, str(p)
    print("    Error: Data directory not found")
    print("    Please ensure there is a 'UNICBench/audio' or 'new_data/audio' folder with annotation files")
    return False, None


def get_available_categories(data_dir_path: str) -> Dict[str, int]:
    from utils.data_loader import get_all_complate_data
    all_data = get_all_complate_data(data_dir_path, data_types=["audio"])
    if not all_data:
        print("   Error: No tasks found in the data directory")
        return {}
    cats: Dict[str, int] = {}
    for t in all_data:
        p = Path(t['file_path'])
        cat = p.parent.name
        cats[cat] = cats.get(cat, 0) + 1
    return cats


def select_categories(data_dir_path: str) -> List[str]:
    cats = get_available_categories(data_dir_path)
    if not cats:
        return []
    print("Discovered categories:")
    keys = list(cats.keys())
    for i, c in enumerate(keys, 1):
        print(f"   {i}. {c} ({cats[c]} tasks)")
    print("\nSelect categories to evaluate:")
    print("   a. All categories")
    print("   or enter category numbers, separated by commas (e.g.: 1,3,5)")
    choice = input("\nPlease select: ").strip().lower()
    if choice == 'a':
        return keys
    try:
        idx = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
        sel = [keys[i] for i in idx if 0 <= i < len(keys)]
        return sel if sel else keys
    except Exception:
        print("   Invalid selection, will evaluate all categories")
        return keys


def main():
    print("   Multi-Model Audio Counting Evaluation System")
    print("=" * 50)

    if HAS_MULTI_MODEL_CONFIG:
        print("\n   Model Selection")
        print("-" * 30)
        selected_model_name = select_model_interactively()
        if not selected_model_name:
            print("   No model selected, exiting...")
            return
        try:
            selected_apis = get_model_config(selected_model_name)
            evaluation_settings = MULTI_MODEL_SETTINGS
            print(f"   Using model: {selected_model_name}")
        except Exception as e:
            print(f"   Error loading model config: {e}")
            print("   Error: No model selected")
            return
    else:
        print("   Error: No model configuration found")
        return

    if not check_api_config(selected_apis):
        return

    ok, data_dir_path = check_data_directory()
    if not ok:
        return

    selected_categories = select_categories(data_dir_path)
    if not selected_categories:
        print("   No categories selected")
        return

    print(f"\n   Will evaluate the following categories: {', '.join(selected_categories)}")

    max_tasks = input(f"\nMaximum number of audios to evaluate per category? (default: {evaluation_settings['max_tasks_per_category']}): ").strip()
    try:
        max_tasks = int(max_tasks) if max_tasks else evaluation_settings['max_tasks_per_category']
    except ValueError:
        max_tasks = evaluation_settings['max_tasks_per_category']
    print(f"  Maximum of {max_tasks} audios per category will be evaluated")

    model_name = input(f"\nEnter model name for results folder (default: {selected_model_name}): ").strip() or selected_model_name

    # Resume support: allow using an existing results directory to continue
    base_model_dir = Path(__file__).parent / 'results' / 'audio_results' / model_name
    resume_mode = False
    resume_choice = input(f"\nEnable resume mode (skip completed tasks)? (y/N): ").strip().lower()
    if resume_choice == 'y':
        resume_mode = True
        print("   Resume mode enabled - will skip already completed tasks")
        print(f"   You may input an existing evaluation directory (e.g. {base_model_dir}/audio_YYYYmmdd_HHMMSS_*)")
        existing_dir_input = input("   Enter existing results directory to resume from (leave empty to create a new one): ").strip()
        if existing_dir_input:
            cand = Path(existing_dir_input)
            if cand.exists() and cand.is_dir():
                run_results_dir = cand
                print(f"   Resuming in existing directory: {run_results_dir}")
            else:
                print("   Warning: Path is invalid or not a directory. A new run directory will be created.")
                run_results_dir = None
        else:
            run_results_dir = None
    else:
        run_results_dir = None

    if run_results_dir is None:
        run_results_dir = create_timestamped_results_dir(selected_categories, data_dir_path, model_name)

    print(f"\n   Results will be saved under: {run_results_dir}")

    print("\n   Initializing evaluator...")
    # Try to pick one sample audio from selected categories for connectivity testing
    sample_audio = None
    try:
        from utils.data_loader import get_all_complate_data
        all_data_tmp = get_all_complate_data(data_dir_path, data_types=["audio"]) or []
        for d in all_data_tmp:
            p = Path(d.get('file_path', ''))
            if p.exists() and p.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg'] and p.parent.name in selected_categories:
                sample_audio = str(p)
                break
    except Exception:
        sample_audio = None

    evaluator = AudioCountingEvaluator(
        apis=selected_apis,
        max_try=evaluation_settings.get("max_try", 6),
        results_dir=str(run_results_dir),
        connectivity_audio_sample=sample_audio
    )

    print("   Generating evaluation tasks...")
    all_tasks = evaluator.generate_tasks_from_audio_dir(data_dir_path)

    filtered = []
    cat_counts: Dict[str, int] = {}
    for t in all_tasks:
        if t.category in selected_categories:
            cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
            if cat_counts[t.category] <= max_tasks:
                filtered.append(t)
    if not filtered:
        print("   No matching tasks found")
        return

    print(f"   Generated {len(filtered)} evaluation tasks")
    print("\n    Task distribution:")
    dist: Dict[str, int] = {}
    for t in filtered:
        dist[t.category] = dist.get(t.category, 0) + 1
    for c, n in dist.items():
        print(f"   {c}: {n} tasks")

    confirm = input(f"\nStart evaluation? (y/N): ").strip().lower()
    if confirm != 'y':
        print("   Evaluation cancelled")
        return

    print("\n   Starting evaluation (by category with intermediate saves)...")
    try:
        results = evaluator.evaluate_by_category(filtered, save_after_each=True, save_intermediate=True, resume=resume_mode)
        report = evaluator.generate_report(results)
        report_path = run_results_dir / "audio_counting_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("\n" + "=" * 50)
        print("   Audio Evaluation Report")
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
        print(f"   Report saved to: {report_path}")
        print(f"\n   Evaluation completed!")
    except KeyboardInterrupt:
        print("\n   Evaluation interrupted by user")
    except Exception as e:
        print(f"\n   Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
