<a href="https://github.com/rongchenggang/UNICBench"><img src="https://img.shields.io/badge/Status-Partially%20Released-orange" alt="Status"></a>
<a href="https://cvpr.thecvf.com/"><img src="https://img.shields.io/badge/Conference-CVPR%202026-blue" alt="Conference"></a>
<a href="https://huggingface.co/datasets/rongchenggang/UNICBench"><img src="https://img.shields.io/badge/🤗%20Dataset-Available-yellow" alt="Dataset"></a>
<a href="https://rongcg5620.github.io/UNICBench-Pages/"><img src="https://img.shields.io/badge/🏆%20Leaderboard-Live-orange" alt="Leaderboard"></a>

# UNICBench: Unified Counting Benchmark for MLLM

Official repository for **"UNICBench: UNIfied Counting Benchmark for MLLM"**.

## 📣 Latest News

* **[2026-03-10]** 🎉 **Evaluation toolkit is now available!** Complete evaluation scripts and model configurations are ready for use.
* **[2026-03-06]** 🎉 Dataset and leaderboard are now live! Check out our [HuggingFace dataset](https://huggingface.co/datasets/rongchenggang/UNICBench) and [leaderboard](https://rongcg5620.github.io/UNICBench-Pages/).
* **[2026-02-25]** Repository created! We are organizing the code, dataset, and evaluation toolkit.
* **[2026-02-21]** Our paper has been accepted to **CVPR 2026**! 🎉

---

## 🔍 Abstract

Counting is a core capability for multimodal large language models (MLLMs), yet there is no unified counting dataset to rigorously evaluate this ability across image, text, and audio. We present UNICBench, a unified multimodal, multi level counting benchmark and evaluation toolkit with accurate ground truth, deterministic numeric parsing, and stratified reporting. The corpus comprises 5,300 images (5,508 QA), 872 documents (5,888 QA), and 2,069 audio clips (2,905 QA), annotated with a three level capability taxonomy and difficulty tags. Under a standardized protocol with fixed splits/prompts/seeds and modality specific matching rules, we evaluate 45 state-of-the-art MLLMs across modalities. Results show strong performance on some basic counting tasks but significant gaps on reasoning and the hardest partitions, highlighting long-tail errors and substantial headroom for improving general counting. UNICBench offers a rigorous and comparable basis for measurement and a public toolkit to accelerate progress.

---

## 🚀 Quick Links

* 📦 **Dataset**: [HuggingFace](https://huggingface.co/datasets/rongchenggang/UNICBench)
* 🏆 **Leaderboard**: [Live Leaderboard](https://rongcg5620.github.io/UNICBench-Pages/)
* 📄 **Paper**: [ArXiv](https://arxiv.org/abs/2603.00595)
* 🛠️ **Evaluation Toolkit**: [UNICBench-Evaluation/](UNICBench-Evaluation/)
* 📧 **Contact**: unicbench@163.com

---

## 🛠️ Features

- **Multi-modal Coverage**: Comprehensive evaluation across Image, Text, and Audio modalities
- **Hierarchical Taxonomy**: Three-level capability classification with difficulty labels
- **Standardized Protocol**: Fixed data splits, prompts, and evaluation metrics
- **Deterministic Parsing**: Accurate numerical extraction and matching rules
- **Public Toolkit**: Evaluation scripts and adapter configurations for 45+ MLLMs
- **Easy Integration**: Simple setup with comprehensive model support

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/rongchenggang/UNICBench.git
cd UNICBench
```

### 2. Download Dataset

Download the dataset from [HuggingFace](https://huggingface.co/datasets/rongchenggang/UNICBench) and place it in `UNICBench-Evaluation/UNICBench/`:

```bash
# The dataset should be organized as:
UNICBench-Evaluation/
└── UNICBench/
    ├── image/          # Image counting data (49 categories)
    ├── text/           # Text counting data (12 categories)  
    └── audio/          # Audio counting data (2 categories)
```

### 3. Install Dependencies

```bash
cd UNICBench-Evaluation
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate unicbench
```

### 4. Configure Your Model

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

### 5. Run Evaluation

```bash
# Image counting
python evaluation/run_image_counting.py

# Text counting  
python evaluation/run_text_counting.py

# Audio counting
python evaluation/run_audio_counting.py
```

The scripts will interactively guide you through model selection, category selection, and task configuration.

---

## 📊 Evaluation Results

Results are automatically saved in `UNICBench-Evaluation/evaluation/results/` with detailed metrics and analysis:

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

---

## 🔧 Supported Models

The toolkit supports 45+ MLLMs including:

- **OpenAI**: GPT-4o, GPT-5, GPT-5-mini
- **Anthropic**: Claude-4.5-sonnet  
- **Google**: Gemini-2.5-pro, Gemini-2.5-flash
- **Open Source**: Qwen2.5-VL, InternVL3, GLM-4.5V, DeepSeek-V3.1

See [UNICBench-Evaluation/docs/model_config_guide.md](UNICBench-Evaluation/docs/model_config_guide.md) for the complete list and configuration details.

---

## 📚 Documentation

- **[Evaluation Guide](UNICBench-Evaluation/docs/evaluation_guide.md)**: Complete evaluation workflow
- **[Model Configuration](UNICBench-Evaluation/docs/model_config_guide.md)**: How to add and configure models  
- **[Label Format](UNICBench-Evaluation/docs/label_format.md)**: Dataset annotation format
- **[Toolkit README](UNICBench-Evaluation/README.md)**: Detailed toolkit usage

---

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Run `python UNICBench-Evaluation/evaluation/test_imports.py` to verify installation
2. **API errors**: Check your API keys in `models_config.py`
3. **Data not found**: Ensure dataset is in `UNICBench-Evaluation/UNICBench/` directory
4. **Permission errors**: Make sure you have write access to the results directory

For more help, see the [Evaluation Guide](UNICBench-Evaluation/docs/evaluation_guide.md).

---

## 🖋️ Citation

If you use our work in your research, please consider citing:

```bibtex
@inproceedings{rong2026unicbench,
  title={UNICBench: UNIfied Counting Benchmark for MLLM},
  author={Rong, Chenggang and Han, Tao and Zhao, Zhiyuan and Fan, Yaowu and Wan, Jia and Guo, Song and Yuan, Yuan and Gao, Junyu},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2026}
}
```

---

## 📮 Contact

For any questions, please submit an issue or contact the authors at unicbench@163.com.

---

## 📜 License

This project is licensed under the MIT License.