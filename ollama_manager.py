"""
ollama_manager.py - Complete Python module for Ollama on Mac Mini M4 Pro (24GB)
Handles model management, chat, streaming, tools, and optimization for your hardware.
"""

import json
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import ollama
from halo import Halo
from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)


# Custom exceptions for better error handling
class OllamaError(Exception):
    """Base exception for Ollama-related errors."""


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to Ollama server."""


class OllamaModelError(OllamaError):
    """Raised when model-related errors occur."""


class OllamaResponseError(OllamaError):
    """Raised when response from Ollama is invalid or malformed."""


class OllamaToolError(OllamaError):
    """Raised when tool execution fails."""


@dataclass
class ChatMessage:
    """Chat message with role and content."""

    role: str
    content: str
    timestamp: Optional[float] = None

    VALID_ROLES = {"user", "assistant", "system", "tool"}

    def __post_init__(self) -> None:
        """Validate ChatMessage after initialization."""
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("ChatMessage role must be a non-empty string")
        if not isinstance(self.content, str):
            raise ValueError("ChatMessage content must be a string")
        if self.role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role: {self.role}. Must be one of: {', '.join(self.VALID_ROLES)}"
            )


class OllamaManager:
    """
    Comprehensive Ollama manager optimized for M4 Pro 24GB Mac Mini.
    Auto-selects best models, manages conversations, tools, and performance.
    """

    # Optimal models for M4 Pro 24GB (7B-21B quantized)
    OPTIMAL_MODELS = [
        "qwen2.5:14b-q5_k_m",  # Best overall (coding/chat) ~35-50 t/s
        "llama3.2:11b-q4_k_s",  # Fast general purpose ~40-60 t/s
        "gemma2:9b-q5_k_m",  # Creative writing ~30-45 t/s
        "deepseek-coder:14b-q4_k_m",  # Coding specialist
        "phi3.5:14b-mini-q4_k_m",  # Fast/light alternative
        "llama3.2:3b",  # Ultra-fast for quick tasks ~80+ t/s
    ]

    DEFAULT_MODEL = "qwen2.5:14b"
    MAX_PROMPT_LENGTH = 100_000
    MAX_TOKENS = 100_000
    MIN_TOKENS = 1
    TEMP_MIN, TEMP_MAX = 0.0, 2.0
    GB_DIVISOR = 1024**3

    def __init__(self, default_model: Optional[str] = None) -> None:
        """Initialize OllamaManager with optional default model."""
        if default_model is not None:
            if not isinstance(default_model, str) or not default_model.strip():
                raise ValueError("default_model must be a non-empty string")

        self.model = default_model or self.DEFAULT_MODEL
        self.conversation_history: List[ChatMessage] = []
        self.available_models = self._get_local_models()
        self.is_streaming = False
        self._check_ollama_running()

    def _check_ollama_running(self) -> None:
        """Verify Ollama server is running."""
        try:
            ollama.list()
            logger.debug("Ollama server connection verified")
        except ConnectionError as e:
            logger.error(f"Ollama server not accessible: {e}")
            raise OllamaConnectionError(
                f"Ollama not running. Start with: `ollama serve` or `brew services start ollama`. "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}") from e

    def _get_local_models(self) -> List[str]:
        """Get locally available models."""
        try:
            models = ollama.list()
            # Handle Pydantic models (ollama returns custom types)
            if hasattr(models, "models"):
                return [m.model for m in models.models if hasattr(m, "model") and m.model]
            # Fallback for dict-like responses
            if isinstance(models, dict) and "models" in models:
                model_list = models.get("models", [])
                if isinstance(model_list, list):
                    return [
                        m.get("name", "") if isinstance(m, dict) else getattr(m, "model", "")
                        for m in model_list
                        if (isinstance(m, dict) and m.get("name")) or (hasattr(m, "model") and m.model)
                    ]
            return []
        except Exception:
            return []

    def _format_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """Format ChatMessage objects for Ollama API (role and content only)."""
        if not messages:
            return []
        if not all(isinstance(m, ChatMessage) for m in messages):
            raise ValueError("All messages must be ChatMessage instances")
        return [{"role": m.role, "content": m.content} for m in messages]

    def _validate_string(self, value: Any, name: str, allow_empty: bool = False) -> str:
        """Validate and return a non-empty string."""
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if not allow_empty and not value.strip():
            raise ValueError(f"{name} cannot be empty")
        return value.strip() if not allow_empty else value

    def _validate_model(self, model: Optional[str]) -> str:
        """Validate and return model name."""
        return self._validate_string(model or self.model, "Model name")

    def _validate_prompt(self, prompt: str) -> str:
        """Validate prompt string."""
        validated = self._validate_string(prompt, "Prompt")
        if len(validated) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt is too long (max {self.MAX_PROMPT_LENGTH:,} characters)")
        return validated

    def _validate_params(self, temperature: float, max_tokens: int) -> tuple[float, int]:
        """Validate temperature and max_tokens."""
        if not isinstance(temperature, (int, float)) or not (
            self.TEMP_MIN <= temperature <= self.TEMP_MAX
        ):
            raise ValueError(
                f"Temperature must be a number between {self.TEMP_MIN} and {self.TEMP_MAX}"
            )
        if not isinstance(max_tokens, int) or not (
            self.MIN_TOKENS <= max_tokens <= self.MAX_TOKENS
        ):
            raise ValueError(
                f"max_tokens must be an integer between {self.MIN_TOKENS} and {self.MAX_TOKENS:,}"
            )
        return temperature, max_tokens

    def _validate_response(self, response: Any) -> Any:
        """Validate response structure and return message object."""
        # Handle Pydantic models (ollama returns custom types)
        if hasattr(response, "message"):
            message = response.message
            if not hasattr(message, "content"):
                raise OllamaResponseError("Response message missing 'content' attribute")
            return message
        # Fallback for dict-like responses
        if not isinstance(response, dict) or "message" not in response:
            raise OllamaResponseError("Invalid response format: missing 'message' field")
        message = response["message"]
        if not isinstance(message, dict):
            raise OllamaResponseError("Response 'message' field must be a dict")
        return message

    def _get_response_content(self, message: Any) -> str:
        """Extract and validate content from message."""
        # Handle Pydantic models
        if hasattr(message, "content"):
            content = message.content
        # Fallback for dict-like
        elif isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = ""
        
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        if not content:
            raise OllamaResponseError("Response message missing 'content' field")
        return content

    def pull_best_model(self, model_tag: Optional[str] = None) -> None:
        """Pull optimal model for M4 Pro 24GB."""
        model = self._validate_string(model_tag or self.DEFAULT_MODEL, "Model name")

        if model not in self.available_models:
            logger.info(f"Pulling model: {model} (optimized for 24GB M4 Pro)")
            spinner = Halo(
                text=f"Pulling {model}...",
                spinner="dots",
                color="cyan",
            )
            spinner.start()
            try:
                ollama.pull(model)
                self.available_models = self._get_local_models()
                if model not in self.available_models:
                    spinner.fail(f"Failed to pull model {model}")
                    raise OllamaModelError(
                        f"Failed to pull model {model}. Model not found after pull."
                    )
                spinner.succeed(f"Successfully pulled {model}")
                logger.success(f"Model {model} pulled successfully")
            except ConnectionError as e:
                spinner.fail(f"Connection error while pulling {model}")
                logger.error(f"Connection error while pulling model: {e}")
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama while pulling model: {e}"
                ) from e
            except Exception as e:
                spinner.fail(f"Failed to pull {model}")
                logger.error(f"Error pulling model {model}: {e}")
                raise OllamaModelError(f"Failed to pull model {model}: {e}") from e
            finally:
                spinner.stop()

        self.model = model
        logger.info(f"Using model: {model}")

    def chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Simple chat interface."""
        prompt = self._validate_prompt(prompt)
        model = self._validate_model(model)
        temperature, max_tokens = self._validate_params(temperature, max_tokens)

        user_msg = ChatMessage("user", prompt)
        messages = self.conversation_history + [user_msg]

        if stream:
            return self._stream_chat(messages, model, temperature, max_tokens, user_msg)

        logger.debug(f"Chat request - model: {model}, prompt length: {len(prompt)}")
        try:
            response: Dict[str, Any] = ollama.chat(  # type: ignore[assignment,no-any-return]
                model=model,
                messages=self._format_messages(messages),
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                },
            )
        except ConnectionError as e:
            logger.error(f"Connection error during chat: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise OllamaError(f"Chat request failed: {e}") from e

        message = self._validate_response(response)
        content = self._get_response_content(message)
        logger.debug(f"Chat response received. Length: {len(content)} chars")

        ai_msg = ChatMessage("assistant", content)
        self.conversation_history.extend([user_msg, ai_msg])
        logger.info(f"Chat response added to history. Total messages: {len(self.conversation_history)}")
        return content

    def _stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float,
        max_tokens: int,
        user_msg: ChatMessage,
    ) -> str:
        """Streaming chat response."""
        self.is_streaming = True
        full_response = ""
        logger.debug(f"Starting stream chat with model: {model}")

        try:
            stream = ollama.chat(  # type: ignore[no-any-return]
                model=model,
                messages=self._format_messages(messages),
                stream=True,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
        except ConnectionError as e:
            self.is_streaming = False
            logger.error(f"Connection error during stream chat: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            self.is_streaming = False
            logger.error(f"Stream chat request failed: {e}")
            raise OllamaError(f"Stream chat request failed: {e}") from e

        try:
            for chunk in stream:
                # Handle both dict and ChatResponse object formats
                if isinstance(chunk, dict):
                    if "error" in chunk:
                        error_msg = chunk.get("error", "Unknown streaming error")
                        self.is_streaming = False
                        logger.error(f"Streaming error: {error_msg}")
                        raise OllamaResponseError(f"Streaming error: {error_msg}")

                    msg = chunk.get("message", {})
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        if isinstance(content, str) and content:
                            print(content, end="", flush=True)
                            full_response += content
                else:
                    # Handle ChatResponse object format
                    try:
                        # Check for error attribute
                        if hasattr(chunk, "error") and chunk.error:
                            self.is_streaming = False
                            logger.error(f"Streaming error: {chunk.error}")
                            raise OllamaResponseError(f"Streaming error: {chunk.error}")

                        # Extract content from ChatResponse.message.content
                        if hasattr(chunk, "message") and chunk.message:
                            if hasattr(chunk.message, "content"):
                                content = chunk.message.content
                                if isinstance(content, str) and content:
                                    print(content, end="", flush=True)
                                    full_response += content
                    except AttributeError:
                        # Skip chunks that don't have expected structure
                        continue

            print()  # New line
            logger.debug(f"Stream completed. Response length: {len(full_response)} chars")
        except KeyboardInterrupt:
            print("\n[Stream interrupted by user]")
            logger.warning("Stream interrupted by user")
            raise
        except Exception as e:
            if isinstance(e, (OllamaError, OllamaResponseError)):
                raise
            logger.error(f"Error during streaming: {e}")
            raise OllamaError(f"Error during streaming: {e}") from e
        finally:
            self.is_streaming = False

        if not full_response.strip():
            logger.error("Received empty response from stream")
            raise OllamaResponseError("Received empty response from stream")

        ai_msg = ChatMessage("assistant", full_response)
        self.conversation_history.extend([user_msg, ai_msg])
        logger.info(f"Chat response added to history. Total messages: {len(self.conversation_history)}")
        return full_response

    def add_tool(self, func: Callable) -> Dict[str, Any]:
        """Convert Python function to Ollama tool format."""
        import inspect

        if not callable(func) or not hasattr(func, "__name__"):
            raise ValueError("Tool must be a callable function with __name__ attribute")

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot inspect function signature: {e}") from e

        empty = inspect.Parameter.empty
        properties = {
            param_name: {
                "type": "number" if param.annotation is int else "string",
                "description": param_name,
            }
            for param_name, param in sig.parameters.items()
        }

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or f"{func.__name__} function",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [
                        p for p, param in sig.parameters.items() if param.default == empty
                    ],
                },
            },
        }

    def chat_with_tools(
        self, prompt: str, tools: Optional[List[Callable]] = None, model: Optional[str] = None
    ) -> str:
        """Chat with tool calling support."""
        prompt = self._validate_prompt(prompt)
        model = self._validate_model(model)

        if tools is not None:
            if not isinstance(tools, list) or not all(callable(t) for t in tools):
                raise ValueError("Tools must be a list of callable functions")

        # Convert tools to tool dicts with error handling
        tool_dicts = []
        for i, tool_func in enumerate(tools or []):
            try:
                tool_dicts.append(self.add_tool(tool_func))
            except Exception as e:
                raise OllamaToolError(
                    f"Failed to convert tool {getattr(tool_func, '__name__', f'#{i}')}: {e}"
                ) from e

        user_msg = ChatMessage("user", prompt)
        messages = self.conversation_history + [user_msg]

        try:
            response: Dict[str, Any] = ollama.chat(  # type: ignore[assignment]
                model=model,
                messages=self._format_messages(messages),
                tools=tool_dicts if tool_dicts else None,
                options={"temperature": 0.1},  # Lower temp for tool accuracy
            )
        except ConnectionError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise OllamaError(f"Chat with tools request failed: {e}") from e

        message = self._validate_response(response)

        # Handle tool calls
        if "tool_calls" in message and tools:
            tool_calls = message.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                raise OllamaResponseError("tool_calls must be a list")

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or "function" not in tool_call:
                    continue

                func_info = tool_call.get("function", {})
                if not isinstance(func_info, dict):
                    continue

                func_name = func_info.get("name", "")
                if not func_name:
                    continue

                # Parse arguments
                args_str = func_info.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    args = args if isinstance(args, dict) else {}
                except json.JSONDecodeError as e:
                    raise OllamaToolError(
                        f"Invalid JSON in tool arguments for {func_name}: {e}"
                    ) from e

                # Find and execute tool
                tool: Optional[Callable] = next(
                    (t for t in tools if getattr(t, "__name__", None) == func_name), None
                )
                if tool is not None:
                    try:
                        result = tool(**args)
                        print(f"Tool {func_name} result: {result}")
                        result_str = json.dumps({"result": result}, default=str)
                    except (TypeError, ValueError):
                        result_str = json.dumps(
                            {"result": str(result), "error": "Could not serialize result"}
                        )
                    except Exception as e:
                        error_msg = f"Tool {func_name} execution failed: {e}"
                        print(f"❌ {error_msg}")
                        result_str = json.dumps({"error": error_msg})
                    self.conversation_history.append(ChatMessage("tool", result_str))
                else:
                    error_msg = f"Tool {func_name} not found in provided tools"
                    print(f"❌ {error_msg}")
                    self.conversation_history.append(
                        ChatMessage("tool", json.dumps({"error": error_msg}))
                    )

        content = self._get_response_content(message)
        ai_msg = ChatMessage("assistant", content)
        self.conversation_history.extend([user_msg, ai_msg])
        return content

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance stats."""
        try:
            models = ollama.list()
            # Handle Pydantic models
            if hasattr(models, "models"):
                total_size = sum(getattr(m, "size", 0) for m in models.models) / self.GB_DIVISOR
            # Fallback for dict-like
            elif isinstance(models, dict) and "models" in models:
                model_list = models.get("models", [])
                total_size = (
                    sum(
                        getattr(m, "size", 0) if hasattr(m, "size") else m.get("size", 0)
                        for m in model_list
                        if hasattr(m, "size") or (isinstance(m, dict) and "size" in m)
                    )
                    / self.GB_DIVISOR
                )
            else:
                total_size = 0.0
        except Exception:
            total_size = 0.0

        return {
            "current_model": self.model,
            "local_models": len(self.available_models),
            "total_model_size_gb": round(total_size, 1),
            "conversation_length": len(self.conversation_history),
            "recommended_models": self.OPTIMAL_MODELS[:3],
            "is_streaming": self.is_streaming,
        }

    def interactive_chat(self) -> None:
        """Interactive chat loop."""
        logger.info("Starting interactive chat session")
        print("🤖 Ollama Chat (M4 Pro 24GB optimized)")
        print(f"Model: {self.model} | Type 'quit', 'clear', 'stats', or 'model <name>'")
        print("-" * 50)

        quit_commands = {"quit", "exit", "q"}
        error_handlers = {
            OllamaConnectionError: lambda e: (
                f"\n❌ Connection error: {e}",
                "💡 Make sure Ollama is running: `ollama serve`",
            ),
            OllamaModelError: lambda e: (f"\n❌ Model error: {e}", None),
            OllamaResponseError: lambda e: (f"\n❌ Response error: {e}", None),
            OllamaError: lambda e: (f"\n❌ Error: {e}", None),
            ValueError: lambda e: (f"\n❌ Invalid input: {e}", None),
        }

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in quit_commands:
                    break
                elif user_input.lower() == "clear":
                    self.clear_history()
                    logger.info("Conversation history cleared")
                    print("✅ Conversation cleared")
                    continue
                elif user_input.lower() == "stats":
                    stats = self.get_stats()
                    logger.debug("Stats requested")
                    print(json.dumps(stats, indent=2))
                    continue
                elif user_input.lower().startswith("model "):
                    new_model = user_input[6:].strip()
                    if not new_model:
                        logger.warning("Empty model name provided")
                        print("❌ Please provide a model name")
                        continue
                    try:
                        if new_model in self.OPTIMAL_MODELS or new_model in self.available_models:
                            self.model = new_model
                            logger.info(f"Model switched to: {new_model}")
                            print(f"✅ Switched to {self.model}")
                        else:
                            logger.warning(f"Model '{new_model}' not found")
                            print(
                                f"❌ Model '{new_model}' not found. Try one of: {', '.join(self.OPTIMAL_MODELS[:3])}"
                            )
                    except Exception as e:
                        logger.error(f"Error switching model: {e}")
                        print(f"❌ Error switching model: {e}")
                    continue

                # Chat
                try:
                    start_time = time.time()
                    logger.info(f"Processing chat request: {user_input[:50]}...")
                    self.chat(user_input, stream=True)
                    duration = time.time() - start_time
                    logger.info(f"Chat response completed in {duration:.2f}s")
                    print(f"\n⏱️  {duration:.1f}s")
                except tuple(error_handlers.keys()) as e:
                    logger.error(f"Chat error: {type(e).__name__}: {e}")
                    msg, hint = error_handlers[type(e)](e)
                    print(msg)
                    if hint:
                        print(hint)

            except KeyboardInterrupt:
                logger.info("Interactive chat session ended by user")
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.exception(f"Unexpected error in interactive chat: {e}")
                print(f"❌ Unexpected error: {e}")
                traceback.print_exc()


# Example tools for your M4 Pro setup
def calculate_tokens(text: str) -> float:
    """Count approximate tokens in text."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    return 0.0 if not text.strip() else len(text.split()) * 1.3


def system_info() -> Dict[str, str]:
    """Get basic system information."""
    return {
        "machine": "Mac Mini M4 Pro",
        "ram": "24GB unified memory",
        "gpu": "16-core GPU",
        "optimized_for": "7B-21B quantized models",
    }


# Quick start example
if __name__ == "__main__":
    try:
        logger.info("Initializing OllamaManager")
        manager = OllamaManager()

        try:
            manager.pull_best_model()
        except OllamaModelError as e:
            logger.warning(f"Model pull failed: {e}. Continuing with available models.")
            print(f"⚠️  Warning: {e}\nContinuing with available models...")

        tools = [calculate_tokens, system_info]
        logger.info("Starting interactive chat session")
        manager.interactive_chat()
    except OllamaConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"❌ Connection error: {e}")
        print("💡 Start Ollama with: `ollama serve` or `brew services start ollama`")
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
