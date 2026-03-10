"""
Evaluators module for different modalities
"""

# 避免导入问题，只在需要时导入
def get_evaluator_classes():
    from image_counting_evaluator import ImageCountingEvaluator
    from text_counting_evaluator import TextCountingEvaluator
    from audio_counting_evaluator import AudioCountingEvaluator
    return ImageCountingEvaluator, TextCountingEvaluator, AudioCountingEvaluator

__all__ = [
    'get_evaluator_classes'
]