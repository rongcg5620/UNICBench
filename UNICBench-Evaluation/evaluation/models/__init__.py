"""
Model interface and configuration module
"""

# 避免循环导入，只在需要时导入
def get_chat_bot_classes():
    from chat_bots import ChatBot, ChatBots
    return ChatBot, ChatBots

def get_model_config_functions():
    from models_config import get_model_config, select_model_interactively
    return get_model_config, select_model_interactively

__all__ = [
    'get_chat_bot_classes',
    'get_model_config_functions'
]