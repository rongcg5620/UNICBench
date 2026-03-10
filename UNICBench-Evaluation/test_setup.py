#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNICBench评测工具包设置测试脚本
验证安装和配置是否正确
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """测试目录结构"""
    print("🔍 检查目录结构...")
    
    current_dir = Path(__file__).parent
    required_dirs = [
        "evaluation",
        "docs", 
        "UNICBench"
    ]
    
    required_files = [
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if not (current_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if not (current_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"❌ 缺少目录: {', '.join(missing_dirs)}")
        return False
    
    if missing_files:
        print(f"❌ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("✅ 目录结构正确")
    return True

def test_data_directory():
    """测试数据目录"""
    print("\n🔍 检查数据目录...")
    
    current_dir = Path(__file__).parent
    data_dir = current_dir / "UNICBench"
    
    if not data_dir.exists():
        print("❌ UNICBench数据目录不存在")
        print("   请从HuggingFace下载数据集: https://huggingface.co/datasets/rongchenggang/UNICBench")
        return False
    
    required_modalities = ["image", "text", "audio"]
    missing_modalities = []
    
    for modality in required_modalities:
        if not (data_dir / modality).exists():
            missing_modalities.append(modality)
    
    if missing_modalities:
        print(f"❌ 缺少数据模态: {', '.join(missing_modalities)}")
        return False
    
    # 检查是否有数据文件
    for modality in required_modalities:
        modality_dir = data_dir / modality
        if not any(modality_dir.iterdir()):
            print(f"❌ {modality}目录为空")
            return False
    
    print("✅ 数据目录结构正确")
    return True

def test_imports():
    """测试导入"""
    print("\n🔍 检查Python导入...")
    
    try:
        # 添加evaluation目录到路径
        eval_dir = Path(__file__).parent / "evaluation"
        sys.path.insert(0, str(eval_dir))
        
        # 测试核心导入
        from evaluators.image_counting_evaluator import ImageCountingEvaluator
        from evaluators.text_counting_evaluator import TextCountingEvaluator  
        from evaluators.audio_counting_evaluator import AudioCountingEvaluator
        from utils.data_loader import get_all_complate_data
        
        print("✅ 核心模块导入成功")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_dependencies():
    """测试依赖包"""
    print("\n🔍 检查依赖包...")
    
    required_packages = [
        "openai",
        "numpy", 
        "pandas",
        "requests",
        "pathlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("   请运行: pip install -r requirements.txt")
        return False
    
    print("✅ 依赖包检查通过")
    return True

def main():
    """主测试函数"""
    print("🚀 UNICBench评测工具包设置测试")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_dependencies,
        test_imports,
        test_data_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！评测工具包已准备就绪。")
        print("\n📝 下一步:")
        print("   1. 配置模型API密钥: evaluation/models/models_config.py")
        print("   2. 运行评测: python evaluation/run_image_counting.py")
        return True
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)