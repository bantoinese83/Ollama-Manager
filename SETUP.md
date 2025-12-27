# Setup Guide - Ollama Manager

## Step 1: Install Ollama

### macOS (using Homebrew - Recommended)
```bash
brew install ollama
```

### Or Download Manually
Visit https://ollama.com/download and download the macOS installer.

## Step 2: Start Ollama Server

### Option A: Start as Service (Recommended)
```bash
brew services start ollama
```

### Option B: Start Manually
```bash
ollama serve
```
Keep this terminal open while using Ollama.

## Step 3: Download Models

The manager is pre-configured with optimal models for your M4 Pro 24GB Mac Mini. You can download them in several ways:

### Option A: Using the Manager (Automatic)
```bash
# Activate virtual environment
source venv/bin/activate

# Run the manager - it will auto-pull the default model
python ollama_manager.py
```

### Option B: Using the Setup Script
```bash
# Activate virtual environment
source venv/bin/activate

# Run setup script
python setup_models.py
```

### Option C: Manual Download via Ollama CLI
```bash
# Download the recommended models
ollama pull qwen2.5:14b-q5_k_m      # Best overall (default)
ollama pull llama3.2:11b-q4_k_s    # Fast general purpose
ollama pull gemma2:9b-q5_k_m        # Creative writing
ollama pull deepseek-coder:14b-q4_k_m  # Coding specialist
ollama pull phi3.5:14b-mini-q4_k_m  # Fast/light alternative
ollama pull llama3.2:3b             # Ultra-fast for quick tasks
```

## Step 4: Verify Installation

```bash
# Check if Ollama is running
ollama list

# Test the manager
python -c "from ollama_manager import OllamaManager; m = OllamaManager(); print('✅ Setup complete!')"
```

## Recommended Models for M4 Pro 24GB

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `qwen2.5:14b-q5_k_m` | ~9GB | 35-50 t/s | **Default** - Best overall (coding/chat) |
| `llama3.2:11b-q4_k_s` | ~7GB | 40-60 t/s | Fast general purpose |
| `gemma2:9b-q5_k_m` | ~6GB | 30-45 t/s | Creative writing |
| `deepseek-coder:14b-q4_k_m` | ~8GB | 30-45 t/s | Coding specialist |
| `phi3.5:14b-mini-q4_k_m` | ~8GB | 35-50 t/s | Fast/light alternative |
| `llama3.2:3b` | ~2GB | 80+ t/s | Ultra-fast for quick tasks |

**Note:** With 24GB unified memory, you can run multiple models simultaneously or keep 2-3 models loaded.

## Troubleshooting

### Ollama not found
```bash
# Check if Ollama is installed
which ollama

# If not found, install via Homebrew
brew install ollama
```

### Connection Error
```bash
# Make sure Ollama is running
brew services list | grep ollama

# If not running, start it
brew services start ollama

# Or start manually
ollama serve
```

### Model Download Fails
- Check internet connection
- Verify Ollama is running: `ollama list`
- Try downloading a smaller model first: `ollama pull llama3.2:3b`

## Next Steps

1. ✅ Install Ollama
2. ✅ Start Ollama server
3. ✅ Download at least one model (recommended: `qwen2.5:14b-q5_k_m`)
4. ✅ Run `python ollama_manager.py` to start interactive chat

Enjoy your optimized Ollama setup! 🚀

