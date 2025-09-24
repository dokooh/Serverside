#!/usr/bin/env python3
"""
Model Preview Script
====================
Shows the list of models that will be tested before running the full benchmark suite.
This helps verify configuration and provides an overview of what will be downloaded/tested.

Usage:
    python preview_models.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_downloader import ModelDownloader
from model_tester import ModelTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def format_size(size_gb: float) -> str:
    """Format size in a human-readable way"""
    if size_gb < 1:
        return f"{size_gb * 1024:.0f} MB"
    else:
        return f"{size_gb:.1f} GB"

def print_banner():
    """Print a nice banner"""
    print("=" * 80)
    print("🤖 HUGGING FACE MODEL BENCHMARK SUITE")
    print("📋 Model Preview & Configuration Summary")
    print("=" * 80)
    print()

def preview_models():
    """Preview all models that will be tested"""
    try:
        # Initialize components
        print("🔧 Initializing model downloader...")
        downloader = ModelDownloader()
        
        print("🔧 Initializing model tester...")
        tester = ModelTester()
        
        print()
        print("📊 CONFIGURED MODELS:")
        print("-" * 50)
        
        total_estimated_size = 0
        model_count = 0
        
        # Display each model configuration
        for model_key, config in downloader.model_configs.items():
            model_count += 1
            print(f"\n{model_count}. 🤖 {model_key.upper()}")
            print(f"   📦 Repository: {config['hf_repo']}")
            print(f"   🏷️  Type: {config.get('type', 'text-generation')}")
            print(f"   📏 Estimated Size: {format_size(config.get('estimated_size_gb', 1.0))}")
            print(f"   ⚙️  Quantization: {'✅ Q2_K preferred' if config.get('prefer_q2k') else '✅ 4-bit preferred' if config.get('quantized_alternatives') else '❌ Full precision only'}")
            
            # Add alternatives if available
            quantized_alts = config.get('quantized_alternatives', [])
            if quantized_alts:
                print(f"   🔄 Quantized Alternatives:")
                for i, alt_repo in enumerate(quantized_alts, 1):
                    print(f"      {i}. {alt_repo}")
            
            total_estimated_size += config.get('estimated_size_gb', 1.0)
        
        print()
        print("=" * 50)
        print(f"📈 SUMMARY:")
        print(f"   • Total Models: {model_count}")
        print(f"   • Estimated Total Size: {format_size(total_estimated_size)}")
        print(f"   • Cache Directory: {downloader.models_dir}")
        print("=" * 50)
        
        print()
        print("🧪 TEST PROMPTS OVERVIEW:")
        print("-" * 30)
        
        # Show prompt categories
        print(f"📝 General Text Prompts: {len(tester.test_prompts)} prompts")
        print(f"👁️  Vision Prompts: {len(tester.vision_prompts)} prompts")
        print(f"📄 Document/OCR Prompts: {len(tester.document_prompts)} prompts")
        print(f"❓ Image QA Prompts: {len(tester.image_qa_prompts)} prompts")
        
        total_text_prompts = len(tester.test_prompts)
        total_vision_prompts = len(tester.vision_prompts) + len(tester.document_prompts) + len(tester.image_qa_prompts)
        
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
        print("   • cache/download_results.json")
        print("   • reports/model_performance_report.html")
        print("   • reports/benchmark_results.json")
        
        print()
        print("🚀 READY TO START!")
        print("=" * 80)
        print("To run the full benchmark suite:")
        print("   python main_orchestrator.py")
        print()
        print("To run individual components:")
        print("   python model_downloader.py    # Download models only")
        print("   python model_tester.py        # Test existing models")
        print("   python report_generator.py    # Generate reports only")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during model preview: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print(f"   • Check if you have sufficient disk space ({format_size(total_estimated_size)} estimated)")
        print(f"   • Verify internet connection for model downloads")
        print(f"   • Error details: {str(e)}")
        return False

def show_detailed_model_info():
    """Show detailed information about each model"""
    print("\n🔍 DETAILED MODEL INFORMATION:")
    print("=" * 60)
    
    downloader = ModelDownloader()
    
    for model_key, config in downloader.model_configs.items():
        print(f"\n📦 {model_key.upper()}:")
        print(f"   Primary: {config['hf_repo']}")
        
        # Show quantized alternatives
        quantized_alts = config.get('quantized_alternatives', [])
        if quantized_alts:
            print(f"   Quantized options: {len(quantized_alts)} available")
            for alt in quantized_alts:
                print(f"     • {alt}")
        
        # Try to get more info about the selected model
        try:
            repo_id = config['hf_repo']
            print(f"   🎯 Selected: {repo_id}")
            
            # Show what type of model files we expect (all are text-generation models now)
            print(f"   📁 Expected files: model weights, tokenizer")
            print(f"   🚀 Capabilities: Text generation, Question answering")
                
        except Exception as e:
            print(f"   ⚠️  Could not resolve best variant: {e}")

def main():
    """Main preview function"""
    print_banner()
    
    # Show basic model preview
    if not preview_models():
        sys.exit(1)
    
    # Ask if user wants detailed info
    print("\n" + "="*50)
    response = input("🔍 Show detailed model information? [y/N]: ").strip().lower()
    if response in ['y', 'yes']:
        show_detailed_model_info()
    
    print("\n🎉 Preview complete! Ready to start benchmarking.")

if __name__ == "__main__":
    main()