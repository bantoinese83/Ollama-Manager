# Ollama Manager for Mac Mini M4 Pro

<div align="center">

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Type Checker](https://img.shields.io/badge/type%20checker-mypy-blue.svg)
![Linter](https://img.shields.io/badge/linter-ruff-orange.svg)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey.svg)

**Complete Python module for managing Ollama on Mac Mini M4 Pro (24GB)**

*Handles model management, chat, streaming, tools, and optimization for your hardware*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Development](#-development) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Architecture](#-architecture)
- [Development](#-development)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Core Features

- **🚀 Hardware-optimized**: Pre-configured for M4 Pro 24GB (7B-21B models, 30-60 t/s)
- **💬 Interactive chat**: Streaming responses with conversation history
- **🔧 Tool calling**: Pass any Python function as a tool for function calling
- **📦 Model management**: Auto-pull best models, list local models
- **🧠 Conversation memory**: Automatic context tracking across sessions
- **📊 Performance monitoring**: Timing, stats, and resource usage tracking

### Developer Features

- **📝 Structured logging**: Powered by [Loguru](https://github.com/Delgan/loguru) with colored output
- **⏳ Visual feedback**: Loading spinners via [Halo](https://github.com/manrajgrover/halo) for long operations
- **🛡️ Type safety**: Full type hints with MyPy validation
- **🎨 Code quality**: Black formatting, Ruff linting, comprehensive error handling
- **🔍 Edge case handling**: Robust validation and custom exception hierarchy

## 🚀 Quick Start

### Prerequisites

- **macOS** (optimized for Mac Mini M4 Pro, but works on other systems)
- **Python 3.11+**
- **Ollama** installed and running
- **24GB+ RAM** (recommended for optimal performance)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/bantoinese83/Ollama-Manager.git
cd Ollama-Manager

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama (if not already installed)
brew install ollama  # macOS
# Or download from https://ollama.com/download

# 5. Start Ollama server
brew services start ollama  # macOS (recommended)
# Or: ollama serve
```

### First Run

```bash
# Download models and start interactive chat
python ollama_manager.py
```

**📖 For detailed setup instructions, see [SETUP.md](SETUP.md)**

## 📦 Installation

### Option 1: From Source (Recommended)

```bash
git clone https://github.com/bantoinese83/Ollama-Manager.git
cd Ollama-Manager
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Development Setup

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Or use Make
make all  # Installs deps and runs all checks
```

## 💻 Usage

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

# Streaming chat (real-time output)
manager.chat("Tell me a story", stream=True)

# Chat with custom parameters
response = manager.chat(
    "Explain quantum computing",
    model="qwen2.5:14b",
    temperature=0.7,
    max_tokens=2048
)
```

### Interactive Chat

```bash
python ollama_manager.py
```

**Interactive Commands:**
- `quit` or `exit` - Exit the chat
- `clear` - Clear conversation history
- `stats` - Show performance stats and model information
- `model <name>` - Switch to a different model

### Advanced Usage

#### Tool Calling

```python
from ollama_manager import OllamaManager

def calculate_tokens(text: str) -> float:
    """Count approximate tokens in text."""
    return len(text.split()) * 1.3

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

manager = OllamaManager()
tools = [calculate_tokens, get_weather]

# Chat with tools
response = manager.chat_with_tools(
    "What's the weather in San Francisco and how many tokens is this question?",
    tools=tools
)
```

#### Model Management

```python
# List available models
manager = OllamaManager()
print(manager.available_models)

# Pull a specific model
manager.pull_best_model("llama3.2:3b")

# Get statistics
stats = manager.get_stats()
print(f"Total models: {stats['local_models']}")
print(f"Total size: {stats['total_model_size_gb']} GB")
print(f"Current model: {stats['current_model']}")

# Clear conversation history
manager.clear_history()
```

#### Streaming with Custom Handling

```python
manager = OllamaManager()

# Streaming returns full response after completion
response = manager.chat(
    "Write a Python function to calculate fibonacci",
    stream=True,  # Shows real-time output
    temperature=0.3,  # Lower temperature for code
    max_tokens=1024
)
```

## 📚 API Documentation

### `OllamaManager`

Main class for managing Ollama interactions.

#### Initialization

```python
manager = OllamaManager(default_model: Optional[str] = None)
```

- `default_model`: Optional model name. Defaults to `qwen2.5:14b`

#### Methods

##### `chat()`

```python
chat(
    prompt: str,
    model: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 4096
) -> str
```

Send a chat message and get a response.

- **Parameters:**
  - `prompt`: User message (max 100,000 chars)
  - `model`: Model name (defaults to instance model)
  - `stream`: Enable streaming output
  - `temperature`: Creativity (0.0-2.0, default 0.7)
  - `max_tokens`: Maximum response length (1-100,000)
- **Returns:** Response text
- **Raises:** `OllamaError`, `OllamaConnectionError`, `OllamaResponseError`

##### `chat_with_tools()`

```python
chat_with_tools(
    prompt: str,
    tools: Optional[List[Callable]] = None,
    model: Optional[str] = None
) -> str
```

Chat with function calling support.

- **Parameters:**
  - `prompt`: User message
  - `tools`: List of callable functions
  - `model`: Model name
- **Returns:** Response text with tool results

##### `pull_best_model()`

```python
pull_best_model(model_tag: Optional[str] = None) -> None
```

Download a model if not already available.

- **Parameters:**
  - `model_tag`: Model name (defaults to `DEFAULT_MODEL`)
- **Raises:** `OllamaModelError`, `OllamaConnectionError`

##### `get_stats()`

```python
get_stats() -> Dict[str, Any]
```

Get performance and model statistics.

- **Returns:** Dictionary with:
  - `current_model`: Currently active model
  - `local_models`: Number of downloaded models
  - `total_model_size_gb`: Total disk usage
  - `conversation_length`: Messages in history
  - `recommended_models`: Top 3 recommended models

##### `clear_history()`

```python
clear_history() -> None
```

Clear conversation history.

##### `add_tool()`

```python
add_tool(func: Callable) -> Dict[str, Any]
```

Convert Python function to Ollama tool format.

- **Parameters:**
  - `func`: Callable function with type hints
- **Returns:** Tool definition dictionary

### Custom Exceptions

- `OllamaError`: Base exception for all Ollama errors
- `OllamaConnectionError`: Connection/server issues
- `OllamaModelError`: Model-related errors
- `OllamaResponseError`: Invalid/malformed responses
- `OllamaToolError`: Tool execution failures

## 🏗️ Architecture

### Design Principles

1. **Type Safety**: Full type hints with MyPy validation
2. **Error Handling**: Custom exception hierarchy for clear error messages
3. **Validation**: Input validation for all user-facing methods
4. **Logging**: Structured logging with Loguru for debugging
5. **Performance**: Optimized for M4 Pro 24GB hardware

### Project Structure

```
monarch-ollama/
├── ollama_manager.py      # Main module
├── setup_models.py        # Interactive model setup script
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml        # Tool configuration (Black, Ruff, MyPy)
├── .ruff.toml           # Ruff linter configuration
├── Makefile             # Development automation
├── check.sh             # Code quality check script
├── README.md            # This file
└── SETUP.md             # Detailed setup guide
```

### Key Components

- **`ChatMessage`**: Dataclass for conversation messages
- **`OllamaManager`**: Main manager class
- **Tool System**: Automatic function-to-tool conversion
- **Streaming**: Real-time response handling
- **Model Management**: Automatic model discovery and pulling

## 🛠️ Development

### Development Setup

```bash
# Clone and setup
git clone https://github.com/bantoinese83/Ollama-Manager.git
cd Ollama-Manager
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Code Quality Tools

This project uses modern Python tooling:

- **Black**: Code formatter (line length: 100)
- **Ruff**: Fast linter with auto-fix capabilities
- **MyPy**: Static type checker
- **Loguru**: Structured logging
- **Halo**: Terminal spinners

### Running Checks

#### Using Make (Recommended)

```bash
make format      # Format code with black
make lint        # Run ruff linter
make lint-fix    # Run ruff and auto-fix issues
make type-check  # Run mypy type checker
make check       # Run all checks (format, lint, type-check)
make all         # Install dev deps and run all checks
make clean       # Remove cache files
```

#### Manual Commands

```bash
# Format code
black ollama_manager.py

# Lint code
ruff check ollama_manager.py

# Auto-fix linting issues
ruff check --fix ollama_manager.py

# Type checking
mypy ollama_manager.py --ignore-missing-imports

# Run all checks
./check.sh
```

### Configuration

Linting and formatting settings are in `pyproject.toml`:

- **Line length**: 100 characters
- **Target Python**: 3.11+
- **Ruff rules**: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PIE, PL, TRY, RUF
- **MyPy**: Strict type checking with `--ignore-missing-imports` for external libs

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test**
   ```bash
   # Run checks before committing
   make check
   ```

3. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new feature"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🧪 Testing

### Manual Testing

```bash
# Test basic functionality
python -c "from ollama_manager import OllamaManager; m = OllamaManager(); print(m.get_stats())"

# Test interactive chat
python ollama_manager.py

# Test model pulling
python -c "from ollama_manager import OllamaManager; m = OllamaManager(); m.pull_best_model('llama3.2:3b')"
```

### Testing Checklist

- [ ] Model initialization
- [ ] Chat (non-streaming)
- [ ] Chat (streaming)
- [ ] Tool calling
- [ ] Model pulling
- [ ] Error handling
- [ ] Interactive chat commands

## 🐛 Troubleshooting

### Common Issues

#### Ollama Server Not Running

```bash
# Check if Ollama is running
ollama list

# Start Ollama
brew services start ollama
# Or: ollama serve
```

**Error**: `OllamaConnectionError: Failed to connect to Ollama`

**Solution**: Ensure Ollama server is running. See [SETUP.md](SETUP.md) for details.

#### Model Not Found

```bash
# List available models
ollama list

# Pull the model
ollama pull qwen2.5:14b
```

**Error**: `OllamaModelError: Model not found`

**Solution**: Pull the model first using `manager.pull_best_model()` or `ollama pull <model>`

#### Empty Stream Response

**Error**: `OllamaResponseError: Received empty response from stream`

**Solution**: This is usually a temporary issue. Try:
1. Check Ollama server logs
2. Restart Ollama server
3. Try a different model
4. Check network connectivity

#### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Error**: `ModuleNotFoundError: No module named 'ollama'`

**Solution**: Install dependencies in activated virtual environment.

### Debug Mode

Enable debug logging:

```python
from loguru import logger
logger.add("debug.log", level="DEBUG")

manager = OllamaManager()
# All operations will be logged to debug.log
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the code style
4. **Run checks** (`make check`)
5. **Commit your changes** (`git commit -m 'feat: Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Code Style

- Follow PEP 8 (enforced by Black and Ruff)
- Use type hints for all functions
- Write docstrings for public methods
- Handle edge cases and validate inputs
- Add logging for important operations

### Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

## 📊 Recommended Models

The module is pre-configured with optimal models for M4 Pro 24GB:

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `qwen2.5:14b` | ~9GB | 35-50 t/s | **Best overall** (coding/chat) - Default |
| `llama3.2:11b-q4_k_s` | ~7GB | 40-60 t/s | Fast general purpose |
| `gemma2:9b-q5_k_m` | ~6GB | 30-45 t/s | Creative writing |
| `deepseek-coder:14b-q4_k_m` | ~8GB | 30-40 t/s | Coding specialist |
| `phi3.5:14b-mini-q4_k_m` | ~8GB | 35-50 t/s | Fast/light alternative |
| `llama3.2:3b` | ~2GB | 80+ t/s | Ultra-fast for quick tasks |

**Note**: Model tags with quantization (e.g., `q5_k_m`) may not be available. Use base model names (e.g., `qwen2.5:14b`).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runtime
- [Loguru](https://github.com/Delgan/loguru) - Structured logging
- [Halo](https://github.com/manrajgrover/halo) - Terminal spinners
- [Black](https://github.com/psf/black) - Code formatting
- [Ruff](https://github.com/astral-sh/ruff) - Fast linting
- [MyPy](https://github.com/python/mypy) - Static type checking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/bantoinese83/Ollama-Manager/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bantoinese83/Ollama-Manager/discussions)

---

<div align="center">

**Made with ❤️ for Mac Mini M4 Pro users**

⭐ Star this repo if you find it useful!

</div>
