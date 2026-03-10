#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试导入是否正常工作
"""

import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    try:
        print("Testing models.models_config...")
        from models.models_config import get_available_model_names
        print(f"✓ Available models: {len(get_available_model_names())} models found")
        
        print("Testing models.chat_bots...")
        from models.chat_bots import ChatBots
        print("✓ ChatBots imported successfully")
        
        print("Testing utils.data_loader...")
        from utils.data_loader import get_all_complate_data
        print("✓ data_loader imported successfully")
        
        print("Testing evaluators...")
        from evaluators.image_counting_evaluator import ImageCountingEvaluator
        from evaluators.text_counting_evaluator import TextCountingEvaluator
        from evaluators.audio_counting_evaluator import AudioCountingEvaluator
        print("✓ All evaluators imported successfully")
        
        print("\n🎉 All imports successful! The toolkit is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports()