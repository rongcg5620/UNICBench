# Unified Counting Benchmark for MLLM - Evaluation Guide

This guide provides step-by-step instructions for using the evaluation toolkit to evaluate Multi-modal Large Language Models (MLLMs) on counting tasks.

## Quick Start

### 1. Installation

#### Option 1: Using pip (Recommended)

```bash
git clone https://github.com/rongchenggang/UNICBench.git
cd UNICBench/UNICBench-Evaluation
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
git clone https://github.com/rongchenggang/UNICBench.git
cd UNICBench/UNICBench-Evaluation
conda env create -f environment.yml
conda activate unicbench
```

#### Verify Installation

```bash
cd evaluation
python test_imports.py
```

The test script will verify all dependencies are correctly installed.

### 2. Prepare Dataset

Place your UNICBench dataset in one of the following locations (checked in priority order):

**Recommended**: Download from [HuggingFace](https://huggingface.co/datasets/rongchenggang/UNICBench) and place in the UNICBench directory:
```
UNICBench-Evaluation/
├── evaluation/
└── UNICBench/
    ├── image/
    │   ├── category1/
    │   └── category2/
    ├── text/
    │   ├── category1/
    │   └── category2/
    └── audio/
        ├── category1/
        └── category2/
```

**Alternative locations**:
- `UNICBench-Evaluation/new_data/{modality}/`
- `../UNICBench/{modality}/` (relative to working directory)
- `../new_data/{modality}/` (relative to working directory)
- `UNICBench/{modality}/` (in current working directory)
- `new_data/{modality}/` (in current working directory)

### 3. Configure Model

Edit `evaluation/models_config.py` to configure your model. See [Model Configuration Guide](model_config_guide.md) for details.

### 4. Run Evaluation

```bash
cd evaluation

# Image counting evaluation
python run_image_counting.py

# Text counting evaluation
python run_text_counting.py

# Audio counting evaluation
python run_audio_counting.py
```

The scripts will guide you through:
- **Model Selection**: Choose from configured models
- **Category Selection**: Select specific categories or evaluate all
- **Task Limits**: Set maximum tasks per category for testing
- **Resume Options**: Continue from interrupted evaluations

### 5. View Results

Results are automatically saved with timestamped directories and comprehensive reports.

## Evaluation Process

### Interactive Workflow

Each evaluation script follows this interactive workflow:

1. **Model Selection**
   - Lists all configured models with descriptions
   - Select by number (1-N) from available options
   - Validates API configuration before proceeding

2. **Data Directory Detection**
   - Automatically searches for data directories in priority order:
     - `../UNICBench/{modality}/` (alongside evaluation directory)
     - `../new_data/{modality}/` (legacy location)
     - `UNICBench/{modality}/` (current directory)
     - `new_data/{modality}/` (current directory)
   - Validates directory structure and annotation files

3. **Category Selection**
   - Discovers available categories from data directory
   - Shows task count per category
   - Options:
     - `a`: Evaluate all categories
     - `1,3,5`: Select specific categories by number

4. **Task Configuration**
   - Set maximum tasks per category (default: 100)
   - Enable/disable batch mode (text only)
   - Enable/disable automatic visualization

5. **Resume Support**
   - Option to resume from existing results directory
   - Automatically skips completed tasks
   - Choice to retry previously failed tasks
   - Rebuilds final results from partial runs

6. **Execution**
   - Processes tasks by category with intermediate saves
   - Shows progress and handles interruptions gracefully
   - Generates comprehensive reports and visualizations

### Result Structure

Results are organized in timestamped directories:

```
evaluation/results/
└── {modality}_results/
    └── {model_name}/
        └── {timestamp}_{label}/
            ├── {category}_results/
            │   ├── {category}_batch_progress_*.json
            │   └── {category}_results_merged.json
            ├── {modality}_counting_report.json
            └── visualization/
                └── comprehensive_report.md
```

### Key Metrics

The evaluation reports include:

- **Extraction Rate**: Percentage of successful predictions
- **Accuracy**: Exact match percentage (when ground truth available)
- **Processing Time**: Average time per task
- **Token Usage**: Input/output tokens for cost analysis
- **Error Analysis**: Categorized failure types

### Result Format

Each result entry contains:

```json
{
    "task": {
        "document_id": "path/to/file",
        "question_id": "q1",
        "ground_truth": 15
    },
    "predicted_count": 14,
    "processing_time": 2.3,
    "total_tokens": 1295,
    "prompt_tokens": 1250,
    "completion_tokens": 45,
    "raw_response": "I can see 14 apples...",
    "error_type": null
}
```

## Advanced Features

### Resume Functionality

The toolkit supports robust resume capabilities:

```bash
# When prompted during evaluation:
Resume from an existing results directory? (y/N): y
Enter existing run dir: evaluation/results/image_results/gpt-4o/20240315_143022_5cats
Retry previously failed tasks? (Y/n): y
```

The system will:
- Skip already completed tasks
- Optionally retry failed tasks
- Maintain task quotas per category
- Rebuild merged result files

### Batch Mode (Text Only)

Text evaluation supports batch processing:

```bash
Enable batch mode for multiple questions per document? (Y/n): y
```

Benefits:
- Process multiple questions per document in single API call
- Reduced API calls and costs
- Improved efficiency for documents with many questions

### Automatic Visualization

Enable automatic visualization generation:

```bash
Generate visualization analysis after evaluation? (Y/n): y
```

Creates comprehensive analysis including:
- Accuracy distributions by category
- Error analysis charts
- Performance metrics dashboard
- Detailed markdown reports

### Custom Configuration

You can modify evaluation behavior by editing the scripts:

```python
# Limit maximum tasks per category
EVALUATION_SETTINGS = {
    "max_tasks_per_category": 50,  # Reduce for testing
    "max_try": 3,                  # API retry attempts
}

# Filter by specific criteria
filtered_tasks = [t for t in all_tasks if meets_criteria(t)]
```

## Troubleshooting

### Common Issues

1. **Model Configuration Errors**
   ```
   Error: No model selected
   ```
   - Ensure `models_config.py` exists and contains valid configurations
   - Check API keys are not placeholder values (no "your-" prefix)
   - Verify API endpoints are accessible

2. **Data Directory Issues**
   ```
   Error: Data directory not found
   ```
   - Ensure data directory exists: `new_data/{modality}/`
   - Check directory contains category subdirectories
   - Verify annotation JSON files are present

3. **API Connection Issues**
   ```
   Error during evaluation: Connection timeout
   ```
   - Check internet connection and API endpoint availability
   - Verify API keys have sufficient credits/quota
   - Consider increasing timeout settings for large files

4. **Memory Issues**
   - Reduce `max_tasks_per_category` for testing
   - Use smaller models (e.g., gpt-4o-mini instead of gpt-4o)
   - Close other applications to free memory

### Resume from Interruption

If evaluation is interrupted:

1. Note the results directory path from console output
2. Restart the evaluation script
3. Choose "Resume from existing directory"
4. Enter the noted directory path
5. The system will continue from where it stopped

### Validation

After evaluation, verify results:

```bash
# Check result files exist
ls evaluation/results/{modality}_results/{model}/{timestamp}/

# Validate JSON format
python -m json.tool {result_file}.json

# Check report metrics
cat {modality}_counting_report.json
```

## Best Practices

### Before Starting

1. **Test Configuration**: Run with 1-2 samples per category first
2. **Verify Data**: Ensure annotation files are properly formatted
3. **Check Quotas**: Verify API rate limits and credit availability
4. **Plan Storage**: Ensure sufficient disk space for results

### During Evaluation

1. **Monitor Progress**: Watch console output for errors or warnings
2. **Save Paths**: Note the results directory path for potential resume
3. **Resource Management**: Monitor system resources and API usage
4. **Backup Results**: Copy important results to secure location

### After Evaluation

1. **Validate Results**: Check result files and report metrics
2. **Review Errors**: Examine failed tasks for patterns
3. **Document Settings**: Keep record of configuration used
4. **Archive Results**: Organize results with clear naming conventions

### Cost Management

1. **Start Small**: Use `max_tasks_per_category` for initial testing
2. **Choose Models Wisely**: Balance cost vs. performance needs
3. **Monitor Tokens**: Track token usage in reports
4. **Use Resume**: Avoid re-running completed tasks

### Performance Optimization

1. **Batch Mode**: Enable for text evaluation when possible
2. **Parallel Processing**: Run different modalities simultaneously
3. **Resource Allocation**: Ensure adequate system resources
4. **Network Stability**: Use stable internet connection for API calls

## Getting Help

- Review this documentation and the [Model Configuration Guide](model_config_guide.md)
- Check console output for specific error messages
- Examine result files for debugging information
- Open an issue on GitHub with detailed error information