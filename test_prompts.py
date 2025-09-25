#!/usr/bin/env python3
"""
Test script to verify the updated prompt configuration
"""
import sys
sys.path.append('.')

from model_tester import ModelTester

def test_prompt_configuration():
    """Test that all prompts are properly configured"""
    print("🧪 Testing Updated Prompt Configuration")
    print("="*60)
    
    # Initialize tester
    tester = ModelTester()
    
    # Test comprehensive prompts
    print(f"📋 Comprehensive Prompts (should be 10):")
    print(f"   Total count: {len(tester.comprehensive_prompts)}")
    print()
    
    # Show each prompt with its category
    for i, prompt in enumerate(tester.comprehensive_prompts, 1):
        category = tester.categorize_prompt(prompt)
        print(f"{i:2d}. [{category}] {prompt}")
    
    print()
    print("="*60)
    
    # Count each category
    categories = {}
    for prompt in tester.comprehensive_prompts:
        category = tester.categorize_prompt(prompt)
        categories[category] = categories.get(category, 0) + 1
    
    print("📊 Category Distribution:")
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count} prompts")
    
    print()
    print(f"✅ Total Categories: {len(categories)}")
    print(f"✅ Total Prompts: {sum(categories.values())}")
    
    # Verify we have all 7 expected categories
    expected_categories = {
        "🔍 Web Search Tool",
        "🧮 Calculation Tool", 
        "🖼️ Image Analysis Tool",
        "📄 OCR",
        "📋 Document Analysis",
        "❓ Image QA",
        "👁️ Vision Description"
    }
    
    actual_categories = set(categories.keys())
    missing = expected_categories - actual_categories
    extra = actual_categories - expected_categories
    
    if missing:
        print(f"⚠️  Missing categories: {missing}")
    if extra:
        print(f"ℹ️  Extra categories: {extra}")
    
    if len(actual_categories) == 7 and not missing:
        print("🎉 All 7 expected prompt types are present!")
    
    return len(tester.comprehensive_prompts) == 10 and len(actual_categories) == 7

if __name__ == "__main__":
    success = test_prompt_configuration()
    if success:
        print("\n✅ Prompt configuration test PASSED!")
    else:
        print("\n❌ Prompt configuration test FAILED!")
        sys.exit(1)