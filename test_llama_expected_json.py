#!/usr/bin/env python3
"""
Test script showing expected JSON responses for Llama-3.2-1B optimized prompts
"""
import sys
sys.path.append('.')

from model_tester import ModelTester

def show_llama_expected_responses():
    """Show expected JSON responses for Llama-3.2-1B prompts"""
    print("🤖 LLAMA-3.2-1B OPTIMIZED PROMPTS & EXPECTED JSON RESPONSES")
    print("="*80)
    
    # Initialize tester
    tester = ModelTester()
    
    # Get Llama-3.2-1B prompts
    prompts = tester.get_model_prompts("llama-3.2-1b")
    system_prompt = tester.get_system_prompt("llama-3.2-1b")
    
    print(f"📋 SYSTEM PROMPT ({len(system_prompt)} chars):")
    print(f'"{system_prompt}"')
    print()
    
    # Expected responses based on your specification
    expected_responses = [
        # 1. Basic transcription
        '{"name":"transcribe_image","arguments":{"image_id":"inv_A12","fields":["date","vendor","total","invoice_no"]}}',
        
        # 2. Web lookup with refinement  
        '{"name":"web_search","arguments":{"query":"ExampleSoft Editor latest stable release"}}',
        
        # 3. Ambiguous screenshot — rule enforcement (assuming >5 words)
        '{"name":"transcribe_image","arguments":{"image_id":"ss_45","fields":["all_text"]}}',
        
        # 4. Chained intent (choose first tool) - likely transcribe first
        '{"name":"transcribe_image","arguments":{"image_id":"img_label_1","fields":["product_name","manufacturer"]}}',
        
        # 5. Numeric compute
        '{"name":"calculate","arguments":{"expression":"PMT(0.035/12,60,50000)"}}',
        
        # 6. Robustness test — truncated prompt
        '{"name":"transcribe_image","arguments":{"image_id":"x9","fields":["date","vendor"]}}',
        
        # 7. Web search test
        '{"name":"web_search","arguments":{"query":"NVIDIA Corporation current stock price"}}',
        
        # 8. Additional transcription
        '{"name":"transcribe_image","arguments":{"image_id":"doc_contract_99","fields":["client_name","contract_date","total_amount"]}}',
        
        # 9. Calculation test
        '{"name":"calculate","arguments":{"expression":"2500 * 0.085"}}',
        
        # 10. Rule enforcement test (assuming receipt type)
        '{"name":"transcribe_image","arguments":{"image_id":"scan_receipt_33","fields":["store","date","total"]}}'
    ]
    
    print("🧪 TEST PROMPTS & EXPECTED JSON RESPONSES:")
    print("-" * 80)
    
    for i, (prompt, expected) in enumerate(zip(prompts, expected_responses), 1):
        category = tester.categorize_prompt(prompt)
        print(f"\n{i:2d}. [{category}]")
        print(f"    User: {prompt}")
        print(f"    Expected: {expected}")
    
    print("\n" + "="*80)
    print("🎯 KEY FEATURES FOR LLAMA-3.2-1B:")
    print("- ✅ 0-shot system prompt (no examples in system)")
    print("- ✅ Rule enforcement with conditional logic")
    print("- ✅ Chained intent handling") 
    print("- ✅ Complex multi-field extraction")
    print("- ✅ Robustness testing with truncated inputs")
    print("- ✅ Financial calculations with PMT function")
    print("- ✅ Best JSON compliance expected among small models")
    
    print("\n🔧 IMPLEMENTATION NOTES:")
    print("- Temperature: 0.1 for deterministic JSON output")
    print("- Validation: JSON structure validation + retry logic")
    print("- Error handling: Malformed response recovery")
    print("- PMT function: Use numeric formula if PMT not supported")
    
    print("\n" + "="*80)
    print("🚀 READY FOR LLAMA-3.2-1B TESTING!")
    print("=" * 80)

if __name__ == "__main__":
    show_llama_expected_responses()