#!/usr/bin/env python3
"""
发布前检查脚本
检查项目是否准备好发布到GitHub
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - 文件不存在")
        return False

def check_directory_structure():
    """检查目录结构"""
    print("检查目录结构...")
    required_dirs = [
        "evaluation",
        "evaluation/models",
        "evaluation/evaluators", 
        "evaluation/utils",
        "docs"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_good = False
    
    return all_good

def check_required_files():
    """检查必需文件"""
    print("\n检查必需文件...")
    required_files = [
        ("README.md", "项目说明文件"),
        ("LICENSE", "许可证文件"),
        ("setup.py", "安装配置文件"),
        ("requirements.txt", "依赖文件"),
        ("requirements-minimal.txt", "精简依赖文件"),
        ("environment.yml", "Conda环境文件"),
        (".gitignore", "Git忽略文件"),
        ("MANIFEST.in", "包含文件清单"),
        ("evaluation/__init__.py", "评测模块初始化"),
        ("evaluation/run_image_counting.py", "图像评测脚本"),
        ("evaluation/run_text_counting.py", "文本评测脚本"),
        ("evaluation/run_audio_counting.py", "音频评测脚本"),
        ("evaluation/test_imports.py", "导入测试脚本"),
        ("docs/evaluation_guide.md", "评测指南"),
        ("docs/model_config_guide.md", "模型配置指南"),
    ]
    
    all_good = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good

def check_sensitive_info():
    """检查是否包含敏感信息"""
    print("\n检查敏感信息...")
    sensitive_patterns = [
        "your-api-key",
        "your-email@domain.com",
        "password",
        "secret"
    ]
    
    # 检查真实的API密钥模式（长度超过20的sk-开头字符串）
    import re
    real_api_key_pattern = re.compile(r'sk-[a-zA-Z0-9]{40,}')
    
    files_to_check = [
        "README.md",
        "setup.py", 
        "evaluation/models/models_config.py",
        "docs/evaluation_guide.md",
        "docs/model_config_guide.md"
    ]
    
    issues_found = False
    for filepath in files_to_check:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                content_lower = content.lower()
                
                # 检查基本敏感模式
                for pattern in sensitive_patterns:
                    if pattern in content_lower:
                        print(f"⚠️  发现敏感信息 '{pattern}' 在文件: {filepath}")
                        issues_found = True
                
                # 检查真实API密钥
                if real_api_key_pattern.search(content):
                    print(f"⚠️  发现真实API密钥在文件: {filepath}")
                    issues_found = True
    
    if not issues_found:
        print("✓ 未发现敏感信息")
    
    return not issues_found

def main():
    """主检查函数"""
    print("🔍 UNICBench-Evaluation 发布前检查")
    print("=" * 50)
    
    checks = [
        ("目录结构", check_directory_structure),
        ("必需文件", check_required_files),
        ("敏感信息", check_sensitive_info),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name}检查失败: {e}")
            all_passed = False
        print()
    
    print("=" * 50)
    if all_passed:
        print("🎉 所有检查通过！项目准备好发布到GitHub。")
        print("\n建议的发布步骤:")
        print("1. git add .")
        print("2. git commit -m 'Initial release of UNICBench-Evaluation toolkit'")
        print("3. git tag v1.0.0")
        print("4. git push origin main --tags")
    else:
        print("❌ 发现问题，请修复后再发布。")
        sys.exit(1)

if __name__ == "__main__":
    main()