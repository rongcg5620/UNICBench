"""
Utility functions and data loading
"""

# 避免导入问题，只在需要时导入
def get_data_loader_functions():
    from data_loader import get_all_complate_data, get_all_json
    return get_all_complate_data, get_all_json

__all__ = [
    'get_data_loader_functions'
]