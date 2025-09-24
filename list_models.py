#!/usr/bin/env python3
"""
Quick Model List
================
Simple script to show just the list of models that will be tested.

Usage:
    python list_models.py
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from model_downloader import ModelDownloader

def main():
    """Show quick model list"""
    print("🤖 Models configured for testing:")
    print("=" * 40)
    
    try:
        downloader = ModelDownloader()
        
        for i, (model_key, config) in enumerate(downloader.model_configs.items(), 1):
            size_info = f"~{config.get('estimated_size_gb', 7.0):.1f}GB"
            model_type = "🔤"  # Text generation model
            print(f"{i}. {model_type} {model_key.upper():<20} ({size_info})")
            print(f"   📦 {config['hf_repo']}")
            print(f"   🎯 Prefers Q2_K: {config.get('prefer_q2k', False)}")
            print()
        
        print(f"Total: {len(downloader.model_configs)} models")
        print("\nRun 'python preview_models.py' for detailed preview")
        print("Run 'python main_orchestrator.py' to start testing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()