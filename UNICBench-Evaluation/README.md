# UNICBench Evaluation Toolkit

Official evaluation toolkit for UNICBench. This folder contains all scripts and tools needed to evaluate models on the UNICBench dataset.

## 📁 Structure

```
UNICBench-Evaluation/
├── UNICBench/              # Dataset directory
│   ├── image/
│   ├── text/
│   └── audio/
├── evaluation/              # Core evaluation scripts
│   ├── run_image_counting.py
│   ├── run_text_counting.py
│   ├── run_audio_counting.py
│   ├── test_imports.py
│   ├── models/             # Model configurations
│   ├── evaluators/         # Evaluation logic
│   └── utils/              # Utility functions
├── docs/                   # Documentation
│   ├── evaluation_guide.md
│   ├── model_config_guide.md
│   └── label_format.md
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── setup.py               # Installation script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the UNICBench-Evaluation directory
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate unicbench
```

### 2. Configure Your Model

Edit `evaluation/models/models_config.py` to add your API credentials:

```python
AVAILABLE_MODELS = {
    "your-model": [
        {
            "type": "OPENAI",
            "base": "https://api.openai.com/v1",
            "key": "sk-your-api-key-here",
            "engine": "gpt-4o",
            "max_tokens": 4096,
            "temperature": 0.0
        }
    ]
}
```

### 3. Run Evaluation

```bash
# Image counting
python evaluation/run_image_counting.py

# Text counting
python evaluation/run_text_counting.py

# Audio counting
python evaluation/run_audio_counting.py
```

The scripts will interactively guide you through:
- Model selection
- Category selection
- Task configuration
- Resume options

## 📊 Results

Evaluation results are saved in `evaluation/results/` with the following structure:

```
results/
├── image_results/
│   └── {model_name}/
│       └── {timestamp}_{categories}/
│           ├── {category}_results/
│           └── image_counting_report.json
├── text_results/
└── audio_results/
```

## 📚 Documentation

- **[Evaluation Guide](docs/evaluation_guide.md)**: Complete evaluation workflow
- **[Model Configuration](docs/model_config_guide.md)**: How to add and configure models
- **[Label Format](docs/label_format.md)**: Dataset annotation format

## 🔧 Supported Models

The toolkit supports 45+ MLLMs including:
- OpenAI: GPT-4o, GPT-5, GPT-5-mini
- Anthropic: Claude-4.5-sonnet
- Google: Gemini-2.5-pro, Gemini-2.5-flash
- Open Source: Qwen2.5-VL, InternVL3, GLM-4.5V, DeepSeek-V3.1

See [docs/model_config_guide.md](docs/model_config_guide.md) for the complete list.

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Run `python evaluation/test_imports.py` to verify installation
2. **API errors**: Check your API keys in `models_config.py`
3. **Data not found**: Ensure dataset is in `UNICBench/` directory

For more help, see the [Evaluation Guide](docs/evaluation_guide.md).

## 📮 Contact

For issues specific to the evaluation toolkit:
- Open an issue in the main repository
- Email: unicbench@163.com
