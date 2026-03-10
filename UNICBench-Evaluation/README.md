# UNICBench-Evaluation

Official evaluation toolkit for Unified Counting Benchmark for MLLM.

## Overview

UNICBench-Evaluation provides a comprehensive evaluation framework for multi-modal counting tasks across image, text, and audio modalities. This toolkit enables researchers to evaluate their models on the Unified Counting Benchmark dataset and compare performance with state-of-the-art MLLMs (Multi-modal Large Language Models).

## Features

- 🖼️ **Image Counting**: Evaluate visual counting capabilities across 49 categories
- 📝 **Text Counting**: Assess text-based counting across 12 categories  
- 🔊 **Audio Counting**: Test audio counting abilities across 2 categories
- 🤖 **Multi-Model Support**: Compatible with GPT-4o, Claude, Gemini, and other LLMs
- 📊 **Comprehensive Analysis**: Built-in metrics, visualization, and comparison tools
- 🔧 **Easy Integration**: Simple API for adding custom models

## Quick Start

### Installation

#### Option 1: Using pip (Recommended)

```bash
git clone https://github.com/your-org/UNICBench-Evaluation.git
cd UNICBench-Evaluation
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
git clone https://github.com/your-org/UNICBench-Evaluation.git
cd UNICBench-Evaluation
conda env create -f environment.yml
conda activate unicbench-evaluation
```

#### Verify Installation

```bash
cd evaluation
python test_imports.py
```

If all imports are successful, you're ready to use the toolkit!

### Basic Usage

1. **Configure your model** in `evaluation/models/models_config.py`:
   ```python
   "your-model": [
       {
           "type": "OPENAI",
           "base": "https://api.openai.com/v1", 
           "key": "sk-your-actual-api-key-here",
           "model": "gpt-4o",
           "max_tokens": 4096,
           "temperature": 0.0
       }
   ]
   ```

2. **Run evaluation**:
   ```bash
   # Image counting
   python evaluation/run_image_counting.py
   
   # Text counting  
   python evaluation/run_text_counting.py
   
   # Audio counting
   python evaluation/run_audio_counting.py
   ```

3. **View results**: Results are automatically saved with comprehensive metrics and can be analyzed using the built-in evaluation reports.

## Directory Structure

```
UNICBench-Evaluation/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── docs/                        # Documentation
│   ├── evaluation_guide.md      # Detailed evaluation guide
│   ├── model_config_guide.md    # Model configuration guide
│   ├── label_format.md          # Label format specification
│   ├── dataset_overview.md      # Dataset overview
│   └── result_format.md         # Result format specification
├── evaluation/                  # Core evaluation code
│   ├── run_image_counting.py    # Image evaluation script
│   ├── run_text_counting.py     # Text evaluation script
│   ├── run_audio_counting.py    # Audio evaluation script
│   ├── evaluators/              # Evaluator implementations
│   ├── models/                  # Model interfaces and configs
│   └── utils/                   # Utility functions
```

## Evaluation Metrics

The toolkit provides comprehensive evaluation metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **ACC**: Accuracy (exact match)
- **Processing Time**: Average response time
- **Token Usage**: Input/output token statistics

## Result Format

Evaluation results are saved in JSON format with the following structure:

```json
{
    "gt_num": 15,                    // Ground truth count
    "pred_num": 14,                  // Model prediction
    "input_token": 1250,             // Input tokens used
    "output_token": 45,              // Output tokens generated
    "raw_response": "{...}", // Full model response
    "error_type": null,              // Error classification (if any)
    "label_path": "apples/001_label.json",
    "question_id": 1,
    "question": "How many apples are in the image?"
}
```

## Supported Models

Currently supported model APIs:
- OpenAI GPT-4o/GPT-4o-mini
- Anthropic Claude
- Google Gemini
- Azure OpenAI
- Custom API endpoints

## Documentation

- [Evaluation Guide](docs/evaluation_guide.md) - Complete usage instructions
- [Model Configuration](docs/model_config_guide.md) - How to configure your models
- [Label Format](docs/label_format.md) - Dataset annotation format reference

## Citation

If you use UNICBench-Evaluation in your research, please cite:

```bibtex
@article{unicbench2024,
  title={Unified Counting Benchmark for MLLM},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## Contact

For questions and support, please contact: [contact@unicbench.org]