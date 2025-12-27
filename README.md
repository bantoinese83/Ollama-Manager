# Ollama Manager for Mac Mini M4 Pro

Complete Python module for managing Ollama on Mac Mini M4 Pro (24GB). Handles model management, chat, streaming, tools, and optimization for your hardware.

## Features

- **Hardware-optimized**: Pre-configured for M4 Pro 24GB (7B-21B models, 30-60 t/s)
- **Interactive chat**: Streaming responses with history/stats
- **Tool calling**: Pass any Python function as a tool
- **Model management**: Auto-pull best models, list local models
- **Conversation memory**: Automatic context tracking
- **Performance monitoring**: Timing and stats

## Quick Start

### 1. Install Ollama
```bash
# macOS (Homebrew)
brew install ollama

# Or download from https://ollama.com/download
```

### 2. Start Ollama Server
```bash
# Start as service (recommended)
brew services start ollama

# Or start manually (keep terminal open)
ollama serve
```

### 3. Setup Python Environment
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Models
```bash
# Option A: Use the setup script (recommended)
python setup_models.py

# Option B: Auto-download when running (will pull default model)
python ollama_manager.py

# Option C: Manual download
ollama pull qwen2.5:14b-q5_k_m  # Default model (~9GB)
```

**📖 For detailed setup instructions, see [SETUP.md](SETUP.md)**

## Usage

### Basic Usage

```python
from ollama_manager import OllamaManager

# Initialize with best model for your hardware
manager = OllamaManager()

# Auto-pull optimal model if needed
manager.pull_best_model()

# Simple chat
response = manager.chat("Hello, how are you?")
print(response)

# Streaming chat
manager.chat("Tell me a story", stream=True)
```

### Interactive Chat

```bash
python ollama_manager.py
```

Commands in interactive mode:
- `quit` or `exit` - Exit the chat
- `clear` - Clear conversation history
- `stats` - Show performance stats
- `model <name>` - Switch to a different model

## Code Quality

This project uses modern Python tooling for code quality:

- **Black**: Code formatter (line length: 100)
- **Ruff**: Fast linter with auto-fix capabilities
- **MyPy**: Static type checker

### Running Checks

Using Make (recommended):
```bash
make format      # Format code with black
make lint        # Run ruff linter
make lint-fix    # Run ruff and auto-fix issues
make type-check  # Run mypy type checker
make check       # Run all checks (format, lint, type-check)
make all         # Install dev deps and run all checks
make clean       # Remove cache files
```

Or manually:
```bash
# Format code
black ollama_manager.py

# Lint code
ruff check ollama_manager.py

# Auto-fix linting issues
ruff check --fix ollama_manager.py

# Type checking
mypy ollama_manager.py --ignore-missing-imports
```

## Recommended Models

The module is pre-configured with optimal models for M4 Pro 24GB:

- `qwen2.5:14b-q5_k_m` - Best overall (coding/chat) ~35-50 t/s (default)
- `llama3.2:11b-q4_k_s` - Fast general purpose ~40-60 t/s
- `gemma2:9b-q5_k_m` - Creative writing ~30-45 t/s
- `deepseek-coder:14b-q4_k_m` - Coding specialist
- `phi3.5:14b-mini-q4_k_m` - Fast/light alternative
- `llama3.2:3b` - Ultra-fast for quick tasks ~80+ t/s

## Configuration

Linting and formatting settings are configured in `pyproject.toml`:
- Line length: 100 characters
- Target Python version: 3.11+
- Ruff rules: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PIE, PL, TRY, RUF

## License

MIT

