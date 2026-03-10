# Model Configuration Guide

This guide explains how to configure different models for the Unified Counting Benchmark for MLLM evaluation toolkit.

## Configuration File

Models are configured in `evaluation/models_config.py` in the `AVAILABLE_MODELS` dictionary.

## Configuration Structure

Each model configuration follows this structure:

```python
AVAILABLE_MODELS = {
    "model-name": [
        {
            "type": "API_TYPE",           # OPENAI, AZURE, or DIRECT
            "base": "API_ENDPOINT",       # API base URL
            "key": "API_KEY",             # Your API key
            "engine": "ENGINE_NAME",      # Model engine/deployment name
            "version": "API_VERSION",     # API version (for Azure)
            "max_tokens": 4096,           # Maximum output tokens
            "temperature": 0.0,           # Sampling temperature
            "comment": "Description"      # Human-readable description
        }
    ]
}
```

## Supported API Types

### Azure OpenAI (`type: "AZURE"`)

For Azure OpenAI deployments:

```python
"gpt4o": [
    {
        "type": "AZURE",
        "base": "https://your-resource.openai.azure.com/",
        "key": "your-azure-api-key-here",
        "engine": "gpt4o",                    # Your Azure deployment name
        "version": "2024-02-15-preview",
        "max_tokens": 4096,
        "temperature": 0.0,
        "use_responses": False,               # Optional: Use Responses API
        "comment": "Azure GPT-4o for image/text/audio counting"
    }
]
```

### OpenAI Compatible APIs (`type: "OPENAI"`)

For OpenAI API or compatible endpoints:

```python
"custom-model": [
    {
        "type": "OPENAI",
        "base": "http://127.0.0.1:8100/v1",   # Local or remote endpoint
        "key": "EMPTY",                       # Some local APIs don't need keys
        "engine": "your-model-name",
        "max_tokens": 4096,
        "temperature": 0.0,
        "timeout_seconds": 1800,              # Optional: Custom timeout
        "comment": "Local model via OpenAI-compatible API"
    }
]
```

### Direct Model Loading (`type: "DIRECT"`)

For direct model loading without API:

```python
"phi-4-14b": [
    {
        "type": "DIRECT",
        "model_path": "/path/to/your/model",
        "engine": "phi-4-14B",
        "max_tokens": 4096,
        "temperature": 0.0,
        "comment": "Direct model loading with transformers"
    }
]
```

## Advanced Configuration Options

### Responses API Support

For models supporting OpenAI's Responses API:

```python
"gpt-5": [
    {
        "type": "AZURE",
        "base": "https://your-resource.openai.azure.com/",
        "key": "your-azure-api-key-here",
        "engine": "gpt-5",
        "version": "2025-04-01-preview",
        "max_tokens": 4096,
        "temperature": 0.0,
        "use_responses": True,              # Enable Responses API
        "reasoning_effort": "minimal",      # For reasoning models: minimal/low/medium/high
        "text_verbosity": "low",           # Output verbosity: low/medium/high
        "comment": "GPT-5 with Responses API"
    }
]
```

### Model-Specific Parameters

Some models require special parameter handling defined in `MODEL_PARAM_SPECS`:

```python
MODEL_PARAM_SPECS = {
    "gpt-5": {
        "supports_custom_temperature": False,        # Model doesn't support custom temperature
        "max_tokens_param": "max_completion_tokens", # Use different parameter name
        "default_temperature": 1.0,                 # Default temperature value
    },
    "o3": {
        "supports_custom_temperature": False,
        "max_tokens_param": "max_completion_tokens",
        "default_temperature": 1.0,
    }
}
```

### Extended Configuration Options

```python
"advanced-model": [
    {
        "type": "OPENAI",
        "base": "https://api.provider.com/v1",
        "key": "sk-your-actual-api-key-here",
        "engine": "model-name",
        "max_tokens": 32768,
        "temperature": 0.0,
        "timeout_seconds": 1800,            # 30 minutes timeout
        "extra_body": {                     # Model-specific parameters
            "thinking": {"type": "disabled"}
        },
        "comment": "Advanced model configuration"
    }
]
```

## Pre-configured Models

The toolkit includes configurations for popular models (replace API keys with your own):

### OpenAI Models
- `gpt4o`: GPT-4o via Azure OpenAI
- `gpt-4o-mini`: GPT-4o-mini via Azure OpenAI  
- `gpt-5`: GPT-5 via Azure OpenAI
- `gpt-5-mini`: GPT-5-mini via Azure OpenAI

### Audio-Specific Models
- `gpt-audio`: GPT-Audio via Azure OpenAI
- `gpt-audio-mini`: GPT-Audio-mini via Azure OpenAI
- `gpt-4o-audio-preview`: GPT-4o Audio Preview

### Open Source Models
- `GLM-4.1V-9B-Thinking`: Via LMDeploy
- `InternVL2_5-78B`: Via LMDeploy
- `Qwen2.5-VL-*`: Via vLLM
- `DeepSeek-V3.1`: Via Azure or DMX API

### Third-Party APIs
- `gemini-2.5-pro`: Via DMX API
- `claude-sonnet-4`: Via DMX API
- `deepseek-v3.1-nothinking`: Via DMX API

## Adding Your Model

### Step 1: Choose Configuration Template

Select the appropriate template based on your API type:

- **Azure OpenAI**: Use `gpt4o` as template
- **OpenAI Compatible**: Use `GLM-4.1V-9B-Thinking` as template  
- **Third-party API**: Use `gemini-2.5-pro` as template
- **Direct Loading**: Use `phi-4-14b` as template

### Step 2: Replace Placeholder Values

**Critical**: Replace all placeholder values with your actual credentials:

```python
"your-model": [
    {
        "type": "AZURE",
        "base": "https://your-actual-resource.openai.azure.com/",  # Your endpoint
        "key": "your-actual-api-key-here",                         # Your API key
        "engine": "your-deployment-name",                          # Your deployment
        "version": "2024-02-15-preview",
        "max_tokens": 4096,
        "temperature": 0.0,
        "comment": "Your model description"
    }
]
```

### Step 3: Test Configuration

Before running full evaluation, test your configuration:

```bash
cd evaluation
python -c "
from models_config import get_model_config, select_model_interactively
config = get_model_config('your-model')
print('Configuration loaded successfully:', config[0]['comment'])
"
```

### Step 4: Verify in Interactive Selection

Run any evaluation script and verify your model appears in the selection menu:

```bash
python run_image_counting.py
# Should show your model in the list with description
```

## Security Best Practices

### API Key Management

1. **Never commit real API keys** to version control
2. **Use environment variables** for sensitive values:

```python
import os

"your-model": [
    {
        "type": "AZURE",
        "base": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "key": os.getenv("AZURE_OPENAI_KEY"),
        "engine": "your-deployment",
        # ... rest of config
    }
]
```

3. **Rotate keys regularly** and monitor usage

### Configuration Validation

The toolkit automatically validates:
- API key format (rejects placeholder values like "sk-your-actual-api-key")
- Endpoint accessibility
- Model availability

## Troubleshooting

### Common Configuration Errors

1. **"No model selected" error**
   - Check `models_config.py` exists and is valid Python
   - Ensure `AVAILABLE_MODELS` dictionary is properly defined

2. **"Please configure your API key" error**
   - Replace placeholder API keys with actual values
   - Remove "your-" prefix from API keys

3. **Connection timeout errors**
   - Verify API endpoint URL is correct
   - Check network connectivity
   - Increase `timeout_seconds` if needed

4. **Model not found errors**
   - Verify `engine` name matches your deployment/model name
   - Check API version compatibility
   - Ensure model supports required modalities (image/text/audio)

### Testing Individual Components

```python
# Test API connectivity
from chat_bots import ChatBots
from models_config import get_model_config

config = get_model_config("your-model")
bot = ChatBots(config)
response = bot.ask("Hello, can you count to 3?")
print(response)
```