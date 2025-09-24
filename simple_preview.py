#!/usr/bin/env python3
"""
Simple Model Preview
====================
Lightweight script to show model configuration without requiring heavy dependencies.

Usage:
    python simple_preview.py
"""

import json
import sys
from pathlib import Path

def format_size(size_gb: float) -> str:
    """Format size in a human-readable way"""
    if size_gb < 1:
        return f"{size_gb * 1024:.0f} MB"
    else:
        return f"{size_gb:.1f} GB"

def get_models_config():
    """Get model configuration directly without importing heavy modules"""
    return {
        "llama-3.2-1b": {
            "primary": "meta-llama/Llama-3.2-1B",
            "quantized_alternatives": [
                "unsloth/Llama-3.2-1B-bnb-4bit",
                "microsoft/Llama-3.2-1B-Instruct-GGUF",
                "bartowski/Llama-3.2-1B-GGUF"
            ],
            "type": "text-generation",
            "estimated_size_gb": 1.2
        },
        "tinyllama": {
            "primary": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantized_alternatives": [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "microsoft/TinyLlama-1.1B-Chat-v1.0-onnx"
            ],
            "type": "text-generation",
            "estimated_size_gb": 0.3
        },
        "kosmos-2": {
            "primary": "microsoft/kosmos-2-patch14-224",
            "quantized_alternatives": [
                "Xenova/kosmos-2-patch14-224"
            ],
            "type": "vision-text-to-text",
            "estimated_size_gb": 0.8
        }
    }

def get_prompt_counts():
    """Get prompt counts without importing model_tester"""
    return {
        "general_text": 8,
        "vision_basic": 3,
        "document_ocr": 10,
        "image_qa": 10
    }

def print_banner():
    """Print a nice banner"""
    print("=" * 80)
    print("🤖 HUGGING FACE MODEL BENCHMARK SUITE")
    print("📋 Model Preview & Configuration Summary")
    print("=" * 80)
    print()

def main():
    """Main preview function"""
    print_banner()
    
    try:
        models_config = get_models_config()
        prompt_counts = get_prompt_counts()
        
        print("📊 CONFIGURED MODELS:")
        print("-" * 50)
        
        total_estimated_size = 0
        model_count = 0
        
        # Display each model configuration
        for model_key, config in models_config.items():
            model_count += 1
            print(f"\n{model_count}. 🤖 {model_key.upper()}")
            print(f"   📦 Repository: {config['primary']}")
            print(f"   🏷️  Type: {config['type']}")
            print(f"   📏 Estimated Size: {format_size(config.get('estimated_size_gb', 1.0))}")
            print(f"   ⚙️  Quantization: {'✅ 4-bit preferred' if config.get('quantized_alternatives') else '❌ Full precision only'}")
            
            # Add alternatives if available
            quantized_alts = config.get('quantized_alternatives', [])
            if quantized_alts:
                print(f"   🔄 Quantized Alternatives ({len(quantized_alts)}):")
                for i, alt_repo in enumerate(quantized_alts, 1):
                    print(f"      {i}. {alt_repo}")
            
            total_estimated_size += config.get('estimated_size_gb', 1.0)
        
        print()
        print("=" * 50)
        print(f"📈 SUMMARY:")
        print(f"   • Total Models: {model_count}")
        print(f"   • Estimated Total Size: {format_size(total_estimated_size)}")
        print(f"   • Cache Directory: ./models")
        print("=" * 50)
        
        print()
        print("🧪 TEST PROMPTS OVERVIEW:")
        print("-" * 30)
        
        # Show prompt categories
        print(f"📝 General Text Prompts: {prompt_counts['general_text']} prompts")
        print(f"👁️  Vision Prompts: {prompt_counts['vision_basic']} prompts")
        print(f"📄 Document/OCR Prompts: {prompt_counts['document_ocr']} prompts")
        print(f"❓ Image QA Prompts: {prompt_counts['image_qa']} prompts")
        
        total_text_prompts = prompt_counts['general_text']
        total_vision_prompts = prompt_counts['vision_basic'] + prompt_counts['document_ocr'] + prompt_counts['image_qa']
        
        print()
        print("🎯 TESTING STRATEGY:")
        print("-" * 20)
        print(f"   • Text-only models: {total_text_prompts} prompts each")
        print(f"   • Vision models: 9 mixed prompts each (vision + OCR + QA)")
        print(f"   • Max prompts per model: 5 (for efficiency)")
        
        print()
        print("📊 BENCHMARK METRICS:")
        print("-" * 20)
        print("   • ⏱️  Response Time (seconds)")
        print("   • 🧠 Memory Usage (MB)")
        print("   • 🔥 GPU Utilization (%)")
        print("   • 📈 Token Generation Rate")
        print("   • 💾 Model Loading Time")
        
        print()
        print("📁 OUTPUT FILES:")
        print("-" * 15)
        print("   • logs/benchmark_YYYYMMDD_HHMMSS.log")
        print("   • models/download_results.json")
        print("   • reports/model_performance_report.html")
        print("   • reports/benchmark_results.json")
        
        print()
        print("🔧 NEXT STEPS:")
        print("-" * 12)
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full benchmark:    python main_orchestrator.py")
        print("3. Or run components individually:")
        print("   • python model_downloader.py    # Download models only")
        print("   • python model_tester.py        # Test existing models")
        print("   • python report_generator.py    # Generate reports only")
        
        print()
        print("🚀 READY TO START!")
        print("=" * 80)
        
        # Show detailed model information
        print("\n🔍 DETAILED MODEL INFORMATION:")
        print("=" * 60)
        
        for model_key, config in models_config.items():
            print(f"\n📦 {model_key.upper()}:")
            print(f"   Primary: {config['primary']}")
            
            # Show quantized alternatives
            quantized_alts = config.get('quantized_alternatives', [])
            if quantized_alts:
                print(f"   Quantized options: {len(quantized_alts)} available")
                for alt in quantized_alts:
                    print(f"     • {alt}")
            
            # Show what type of model files we expect
            if config['type'] == 'vision-text-to-text':
                print(f"   📁 Expected files: model weights, tokenizer, image processor")
                print(f"   🚀 Capabilities: Text + Vision understanding, Image captioning, Visual Q&A")
                print(f"   🖼️  Will test with: Nature images, Document samples, Visual question answering")
            else:
                print(f"   📁 Expected files: model weights, tokenizer")
                print(f"   🚀 Capabilities: Text generation, Question answering")
                print(f"   📝 Will test with: General knowledge, Creative writing, Technical questions")
        
        print("\n🎉 Preview complete! Ready to start benchmarking.")
        return True
        
    except Exception as e:
        print(f"❌ Error during model preview: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   • Make sure you're in the correct directory")
        print(f"   • Check if you have sufficient disk space")
        print(f"   • Verify internet connection for model downloads")
        print(f"   • Error details: {str(e)}")
        return False

if __name__ == "__main__":
    main()