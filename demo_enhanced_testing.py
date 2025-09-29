#!/usr/bin/env python3
"""
Enhanced Model Testing Demo
Shows the new comprehensive 10-prompt testing with all 7 categories
"""
import sys
sys.path.append('.')

from model_tester import ModelTester

def demo_enhanced_testing():
    """Demonstrate the enhanced testing configuration"""
    print("🚀 ENHANCED MODEL TESTING CONFIGURATION")
    print("="*80)
    
    # Initialize tester
    tester = ModelTester()
    
    print(f"📊 TESTING OVERVIEW:")
    print(f"   • Models per test session: ALL available models")
    print(f"   • Model-specific optimization: Llama-3.2-1B & TinyLlama prompts tailored")
    print(f"   • Prompts per model: 10 optimized prompts each (increased from 5)")
    print(f"   • JSON tool calling: All prompts expect JSON responses")
    print(f"   • System prompts: Model-specific with few-shot examples")
    print(f"   • Total test combinations: Models × 10 optimized prompts")
    print()
    
    print(f"🧪 COMPREHENSIVE PROMPT CATEGORIES:")
    print("-" * 60)
    
    # Show optimized prompts for both models
    print("🤖 LLAMA-3.2-1B OPTIMIZED PROMPTS:")
    llama_prompts = tester.get_model_prompts("llama-3.2-1b")
    llama_system = tester.get_system_prompt("llama-3.2-1b")
    
    categories = {}
    for prompt in llama_prompts:
        category = tester.categorize_prompt(prompt)
        if category not in categories:
            categories[category] = []
        categories[category].append(prompt)
    
    print(f"   System prompt: {len(llama_system)} chars - JSON tool calling with rule enforcement")
    
    for i, (category, prompts) in enumerate(sorted(categories.items()), 1):
        print(f"   {i}. {category} ({len(prompts)} prompts)")
        for j, prompt in enumerate(prompts, 1):
            print(f"      {j}. {prompt[:70]}...")
    
    print("\n🤖 TINYLLAMA OPTIMIZED PROMPTS:")
    tiny_prompts = tester.get_model_prompts("tinyllama")
    tiny_system = tester.get_system_prompt("tinyllama")
    
    categories = {}
    for prompt in tiny_prompts:
        category = tester.categorize_prompt(prompt)
        if category not in categories:
            categories[category] = []
        categories[category].append(prompt)
    
    print(f"   System prompt: {len(tiny_system)} chars - Few-shot examples with simple tasks")
    
    # Display each category with examples
    for i, (category, prompts) in enumerate(sorted(categories.items()), 1):
        print(f"\n{i}. {category} ({len(prompts)} prompts)")
        for j, prompt in enumerate(prompts, 1):
            print(f"   {j}. {prompt}")
    
    print("\n" + "="*80)
    print("🎯 ENHANCED TESTING BENEFITS:")
    print("-" * 60)
    print("✅ Comprehensive Coverage: All 7 tool/task types tested")
    print("✅ Consistent Testing: Same 10 prompts for every model")  
    print("✅ Better Comparison: Standardized prompt set enables fair model comparison")
    print("✅ Tool Recognition: Tests model ability to identify needed tools")
    print("✅ Multi-modal Ready: Supports both text-only and vision models")
    print("✅ Category Analysis: Results grouped by prompt type for detailed insights")
    print("✅ Scalable Framework: Easy to add new prompt types or models")
    
    print("\n" + "="*80)
    print("📈 BENCHMARK METRICS PER MODEL:")
    print("-" * 60)
    print("• Performance across all 7 tool categories")
    print("• Average response time per category")
    print("• Token generation speed per prompt type")
    print("• Memory usage patterns for different prompt types")
    print("• Success rate by category (error analysis)")
    print("• Model specialization identification")
    
    print("\n" + "="*80)
    print("🔥 READY FOR ENHANCED TESTING!")
    print("Run: python main_orchestrator.py --hf-token YOUR_TOKEN")
    print("=" * 80)

if __name__ == "__main__":
    demo_enhanced_testing()