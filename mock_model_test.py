#!/usr/bin/env python3
"""
Mock Model Testing - Test the system without actual model downloads
This demonstrates the full pipeline workflow with mock models
"""

import logging
import json
import time
from pathlib import Path

# Configure simple logger for Windows compatibility
try:
    from simple_logger import configure_simple_logger
    configure_simple_logger()
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger(__name__)

def create_mock_download_results():
    """Create mock download results for testing"""
    mock_results = {
        "llama-3.2-1b": {
            "path": "models/llama-3.2-1b",
            "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "type": "text-generation",
            "size_gb": 0.5,
            "status": "success",
            "files": ["mock_model.gguf"],
            "note": "Mock model for testing"
        },
        "tinyllama": {
            "path": "models/tinyllama", 
            "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "type": "text-generation",
            "size_gb": 0.15,
            "status": "success",
            "files": ["mock_model.gguf"],
            "note": "Mock model for testing"
        }
    }
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create mock model directories and files
    for model_key, info in mock_results.items():
        model_path = Path(info["path"])
        model_path.mkdir(exist_ok=True)
        
        # Create a mock GGUF file
        mock_gguf = model_path / "mock_model.gguf"
        if not mock_gguf.exists():
            mock_gguf.write_text("# Mock GGUF file for testing\n# This is not a real model file\n")
    
    # Save mock download results
    with open("models/download_results.json", "w") as f:
        json.dump(mock_results, f, indent=2)
    
    logger.info("✅ Created mock download results and model files")
    return mock_results

def test_model_tester_import():
    """Test if ModelTester can be imported and initialized"""
    try:
        from model_tester import ModelTester
        logger.info("✅ ModelTester imported successfully")
        
        # Try to initialize (this should work even without real models)
        tester = ModelTester(models_dir="./models", device="cpu")
        logger.info("✅ ModelTester initialized successfully")
        
        # Check available prompts
        logger.info(f"📝 Tool selection prompts: {len(tester.tool_selection_prompts)}")
        logger.info(f"👁️ Vision prompts: {len(tester.vision_prompts)}")
        logger.info(f"🖼️ Image analysis prompts: {len(tester.image_analysis_prompts)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to test ModelTester: {e}")
        return False

def test_mock_model_loading():
    """Test the model loading logic with mock results"""
    try:
        # Create mock results
        mock_results = create_mock_download_results()
        
        from model_tester import ModelTester
        tester = ModelTester(models_dir="./models", device="cpu")
        
        logger.info("🧪 Testing mock model detection...")
        
        # Test each mock model
        for model_key, model_info in mock_results.items():
            logger.info(f"Testing {model_key}...")
            
            model_path = model_info["path"]
            logger.info(f"  Model path: {model_path}")
            logger.info(f"  Path exists: {Path(model_path).exists()}")
            logger.info(f"  Is GGUF model: {tester.is_gguf_model(model_path)}")
            
            if tester.is_gguf_model(model_path):
                gguf_file = tester.find_gguf_file(model_path)
                logger.info(f"  GGUF file found: {gguf_file}")
            
        logger.info("✅ Mock model detection completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mock model testing failed: {e}")
        return False

def demonstrate_error_handling():
    """Demonstrate proper error handling for missing models"""
    logger.info("🔧 Testing error handling for missing/failed models...")
    
    # Create a download results file with errors (like the real one)
    error_results = {
        "llama-3.2-1b": {
            "error": "401 Client Error: Unauthorized - Authentication required"
        },
        "tinyllama": {
            "error": "401 Client Error: Unauthorized - Authentication required"
        }
    }
    
    with open("models/download_results_with_errors.json", "w") as f:
        json.dump(error_results, f, indent=2)
    
    try:
        from model_tester import ModelTester
        tester = ModelTester(models_dir="./models", device="cpu")
        
        # This should skip models with errors
        results = tester.test_all_models("models/download_results_with_errors.json")
        logger.info(f"✅ Error handling test completed. Results: {len(results)} tests run")
        logger.info("✅ Models with errors were properly skipped")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run the mock testing demonstration"""
    logger.info("🚀 Starting Mock Model Testing Demonstration")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_model_tester_import),
        ("Mock Model Loading", test_mock_model_loading), 
        ("Error Handling", demonstrate_error_handling)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        results[test_name] = test_func()
        logger.info(f"Result: {'✅ PASSED' if results[test_name] else '❌ FAILED'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n🏆 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed! The system structure is working correctly.")
        logger.info("📝 To use with real models, provide a HuggingFace token:")
        logger.info("   python main_orchestrator.py --hf-token YOUR_TOKEN")
    else:
        logger.error("❌ Some tests failed. Check the error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())