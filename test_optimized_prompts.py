#!/usr/bin/env python3
"""
Test script to show optimized prompts for Llama-3.2-1B and TinyLlama
"""
import sys
sys.path.append('.')

from model_tester import ModelTester

def test_optimized_prompts():
    """Test the optimized prompt configuration for each model"""
    print("🚀 OPTIMIZED MODEL PROMPTS CONFIGURATION")
    print("="*80)
    
    # Initialize tester
    tester = ModelTester()
    
    models = ["llama-3.2-1b", "tinyllama"]
    
    for model in models:
        print(f"\n🤖 MODEL: {model.upper()}")
        print("-" * 60)
        
        # Get system prompt
        system_prompt = tester.get_system_prompt(model)
        print(f"📋 SYSTEM PROMPT ({len(system_prompt)} chars):")
        print(f"   {system_prompt[:150]}...")
        print()
        
        # Get optimized prompts
        prompts = tester.get_model_prompts(model)
        print(f"🧪 OPTIMIZED TEST PROMPTS ({len(prompts)} total):")
        
        # Group prompts by category
        categories = {}
        for i, prompt in enumerate(prompts, 1):
            category = tester.categorize_prompt(prompt)
            if category not in categories:
                categories[category] = []
            categories[category].append((i, prompt))
        
        # Display by category
        for category, prompt_list in sorted(categories.items()):
            print(f"\n   {category} ({len(prompt_list)} prompts):")
            for num, prompt in prompt_list:
                print(f"      {num:2d}. {prompt}")
        
        print("\n" + "="*60)
    
    print("\n🎯 KEY OPTIMIZATIONS:")
    print("-" * 60)
    print("✅ Llama-3.2-1B: Complex prompts with rule enforcement and chained intents")
    print("✅ TinyLlama: Simple, direct prompts with binary decision rules")
    print("✅ Both models: JSON tool calling format with specific examples")
    print("✅ System prompts: Model-specific with few-shot examples")
    print("✅ Temperature: Lower settings for deterministic JSON output")
    
    print("\n🔧 EXPECTED JSON RESPONSES:")
    print("-" * 60)
    print('📝 Image transcription: {"name":"transcribe_image","arguments":{"image_id":"img_001","fields":["date","vendor"]}}')
    print('🔍 Web search: {"name":"web_search","arguments":{"query":"ExampleSoft Editor release date"}}')
    print('🧮 Calculation: {"name":"calculate","arguments":{"expression":"1234.50 * 0.20"}}')
    
    print("\n" + "="*80)
    print("🚀 READY FOR OPTIMIZED TESTING!")
    print("=" * 80)

if __name__ == "__main__":
    test_optimized_prompts()