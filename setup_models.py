#!/usr/bin/env python3
"""
Setup script to download recommended models for M4 Pro 24GB Mac Mini.
"""

import sys
from ollama_manager import OllamaManager, OllamaConnectionError, OllamaModelError

def main():
    """Download recommended models."""
    print("🚀 Ollama Model Setup for M4 Pro 24GB")
    print("=" * 50)
    
    try:
        manager = OllamaManager()
        print(f"✅ Connected to Ollama server")
        print(f"📋 Current models: {len(manager.available_models)}")
        if manager.available_models:
            print(f"   {', '.join(manager.available_models[:3])}...")
        print()
    except OllamaConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   brew services start ollama")
        print("   # or")
        print("   ollama serve")
        sys.exit(1)
    
    # Recommended models in order of priority
    recommended = [
        ("qwen2.5:14b-q5_k_m", "Best overall (coding/chat) - Default"),
        ("llama3.2:11b-q4_k_s", "Fast general purpose"),
        ("gemma2:9b-q5_k_m", "Creative writing"),
    ]
    
    print("📦 Recommended models to download:")
    for i, (model, desc) in enumerate(recommended, 1):
        status = "✅" if model in manager.available_models else "⬜"
        print(f"   {status} {i}. {model}")
        print(f"      {desc}")
    print()
    
    # Ask user which models to download
    print("Which models would you like to download?")
    print("  1. Download all recommended (3 models, ~22GB)")
    print("  2. Download default only (qwen2.5:14b-q5_k_m, ~9GB)")
    print("  3. Choose specific models")
    print("  4. Skip (use existing models)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    models_to_download = []
    
    if choice == "1":
        models_to_download = [model for model, _ in recommended]
    elif choice == "2":
        models_to_download = [recommended[0][0]]
    elif choice == "3":
        print("\nAvailable models:")
        for i, (model, desc) in enumerate(recommended, 1):
            if model not in manager.available_models:
                print(f"  {i}. {model} - {desc}")
        selections = input("Enter model numbers (comma-separated, e.g., 1,2): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selections.split(",")]
            models_to_download = [recommended[i][0] for i in indices if 0 <= i < len(recommended)]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            sys.exit(1)
    elif choice == "4":
        print("✅ Skipping download. Using existing models.")
        if manager.available_models:
            print(f"Available: {', '.join(manager.available_models)}")
        else:
            print("⚠️  No models available. You'll need to download at least one.")
        sys.exit(0)
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    # Filter out already downloaded models
    models_to_download = [m for m in models_to_download if m not in manager.available_models]
    
    if not models_to_download:
        print("✅ All selected models are already downloaded!")
        sys.exit(0)
    
    # Download models
    print(f"\n📥 Downloading {len(models_to_download)} model(s)...")
    print("   This may take a while depending on your internet speed.\n")
    
    for model in models_to_download:
        try:
            print(f"⬇️  Downloading {model}...")
            manager.pull_best_model(model)
            print(f"✅ Successfully downloaded {model}\n")
        except OllamaModelError as e:
            print(f"❌ Failed to download {model}: {e}\n")
        except KeyboardInterrupt:
            print(f"\n⚠️  Download interrupted. {model} may be partially downloaded.")
            print("   You can resume later by running this script again.")
            sys.exit(1)
    
    print("🎉 Setup complete!")
    print(f"\n📊 Final stats:")
    stats = manager.get_stats()
    print(f"   Models: {stats['local_models']}")
    print(f"   Total size: {stats['total_model_size_gb']} GB")
    print(f"   Current model: {stats['current_model']}")
    print("\n🚀 Ready to use! Run: python ollama_manager.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

