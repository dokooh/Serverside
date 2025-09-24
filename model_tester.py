#!/usr/bin/env python3
"""
Model Benchmark Testing Script
Tests different models with various prompts and measures performance
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    TextStreamer, BitsAndBytesConfig, pipeline
)
from PIL import Image
import requests
import gc

from resource_monitor import BenchmarkRunner, BenchmarkResult

logger = logging.getLogger(__name__)

# Try to import GGUF support (optional)
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
    logger.debug("GGUF support available via llama-cpp-python")
except ImportError:
    GGUF_AVAILABLE = False
    logger.debug("GGUF support not available - will use standard transformers")

class ModelTester:
    """Tests individual model performance with various prompts"""
    
    def __init__(self, models_dir: str = "./models", device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark_runner = BenchmarkRunner(sampling_interval=0.1)
        
        # Sample prompts for testing (web-search style)
        self.test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do you cook pasta properly?",
            "What are the benefits of renewable energy?",
            "Write a short story about a robot discovering emotions.",
            "List the top 5 programming languages in 2024.",
            "Describe the process of photosynthesis.",
            "What is the difference between AI and machine learning?"
        ]
        
        # Vision prompts for vision-capable models
        self.vision_prompts = [
            "Describe what you see in this image.",
            "What objects are present in this image?",
            "What is the main subject of this image?"
        ]
        
        # Sample image URL for vision models
        self.sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.loaded_processors = {}
    
    def get_quantization_config(self) -> BitsAndBytesConfig:
        """Get 4-bit quantization config for memory efficiency"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def is_gguf_model(self, model_path: str) -> bool:
        """Check if model directory contains GGUF files"""
        model_dir = Path(model_path)
        return any(f.suffix.lower() == '.gguf' for f in model_dir.rglob('*.gguf'))
    
    def find_gguf_file(self, model_path: str, prefer_q4_k_m: bool = True) -> Optional[str]:
        """Find the best GGUF file in the model directory"""
        model_dir = Path(model_path)
        gguf_files = list(model_dir.rglob('*.gguf'))
        
        if not gguf_files:
            return None
        
        # Prefer Q4_K_M if available and requested
        if prefer_q4_k_m:
            for gguf_file in gguf_files:
                if 'Q4_K_M' in gguf_file.name:
                    logger.info(f"Found preferred Q4_K_M GGUF file: {gguf_file}")
                    return str(gguf_file)
        
        # Otherwise, return the first GGUF file found
        logger.info(f"Using GGUF file: {gguf_files[0]}")
        return str(gguf_files[0])
    
    def load_gguf_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a GGUF model using llama-cpp-python"""
        if not GGUF_AVAILABLE:
            logger.warning(f"GGUF support not available for {model_key}, falling back to transformers")
            return self.load_vision_model_fallback(model_path, model_key)
        
        logger.info(f"Loading GGUF model: {model_key}")
        
        gguf_file = self.find_gguf_file(model_path)
        if not gguf_file:
            raise FileNotFoundError(f"No GGUF file found in {model_path}")
        
        try:
            # Load GGUF model with llama-cpp-python
            logger.info(f"Loading GGUF file: {gguf_file}")
            
            model = Llama(
                model_path=gguf_file,
                n_ctx=2048,  # Context length
                n_gpu_layers=-1 if self.device == "cuda" else 0,  # Use GPU if available
                verbose=False
            )
            
            # Create a simple processor-like object for compatibility
            class SimpleProcessor:
                def __init__(self):
                    pass
                
                def decode(self, tokens, skip_special_tokens=True):
                    # For GGUF models, this will be handled differently
                    return ""
            
            processor = SimpleProcessor()
            
            self.loaded_models[model_key] = model
            self.loaded_processors[model_key] = processor
            
            logger.info(f"✓ GGUF model {model_key} loaded successfully")
            return model, processor
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model {model_key}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {e}")
            raise
    
    def load_vision_model_fallback(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Fallback method for loading vision models when GGUF is not available"""
        logger.warning(f"Using fallback loading for {model_key}")
        # This would use the standard transformers approach
        # For now, we'll raise an error to indicate GGUF is needed
        raise NotImplementedError(f"GGUF model {model_key} requires llama-cpp-python installation")
    
    def load_text_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a text generation model"""
        logger.info(f"Loading text model: {model_key}")
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Device: {self.device}")
        
        try:
            # Try to load tokenizer
            logger.debug(f"Loading tokenizer for {model_key}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            logger.debug(f"✓ Tokenizer loaded successfully for {model_key}")
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug(f"✓ Added pad token for {model_key}")
            
            # Load model with quantization if needed
            logger.debug(f"Preparing model configuration for {model_key}...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                logger.debug(f"Using CUDA configuration with quantization for {model_key}")
                model_kwargs["quantization_config"] = self.get_quantization_config()
                model_kwargs["device_map"] = "auto"
            else:
                logger.debug(f"Using CPU configuration for {model_key}")
            
            logger.debug(f"Model kwargs: {model_kwargs}")
            logger.info(f"Loading model weights for {model_key}... (this may take a while)")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info(f"✓ Model weights loaded successfully for {model_key}")
            
            if self.device == "cpu":
                logger.debug(f"Moving model to CPU for {model_key}")
                model = model.to(self.device)
            
            # Store loaded components
            self.loaded_models[model_key] = model
            self.loaded_tokenizers[model_key] = tokenizer
            
            logger.info(f"✓ Text model {model_key} loaded and ready for inference")
            logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load text model {model_key}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {e}")
            raise
    
    def load_vision_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a vision-language model"""
        logger.info(f"Loading vision model: {model_key}")
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Device: {self.device}")
        
        # Check if this is a GGUF model
        if self.is_gguf_model(model_path):
            logger.info(f"Detected GGUF model for {model_key}")
            return self.load_gguf_model(model_path, model_key)
        
        try:
            logger.debug(f"Loading processor for vision model {model_key}...")
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            logger.debug(f"✓ Processor loaded successfully for {model_key}")
            
            logger.debug(f"Preparing vision model configuration for {model_key}...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                logger.debug(f"Using CUDA configuration for vision model {model_key}")
                model_kwargs["device_map"] = "auto"
            else:
                logger.debug(f"Using CPU configuration for vision model {model_key}")
            
            logger.debug(f"Vision model kwargs: {model_kwargs}")
            logger.info(f"Loading vision model weights for {model_key}... (this may take a while)")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info(f"✓ Vision model weights loaded successfully for {model_key}")
            
            if self.device == "cpu":
                logger.debug(f"Moving vision model to CPU for {model_key}")
                model = model.to(self.device)
            
            # Store loaded components
            self.loaded_models[model_key] = model
            self.loaded_processors[model_key] = processor
            
            logger.info(f"✓ Vision model {model_key} loaded and ready for inference")
            logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, processor
            
        except Exception as e:
            logger.error(f"❌ Failed to load vision model {model_key}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {e}")
            raise
    
    def generate_text_response(self, model_key: str, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate text response from a text model"""
        logger.debug(f"Generating text response for {model_key}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        logger.debug(f"Max new tokens: {max_new_tokens}")
        
        if model_key not in self.loaded_models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.loaded_models[model_key]
        tokenizer = self.loaded_tokenizers[model_key]
        logger.debug(f"Using model device: {model.device}")
        
        # Prepare input
        logger.debug(f"Tokenizing input for {model_key}...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_length = inputs['input_ids'].shape[1]
        logger.debug(f"Input tokens: {input_length}")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.debug(f"Input moved to device: {model.device}")
        
        # Generate
        logger.debug(f"Starting text generation for {model_key}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        output_length = outputs[0].shape[0]
        new_tokens = output_length - input_length
        logger.debug(f"Generated {new_tokens} new tokens for {model_key}")
        
        # Decode response (only the new tokens)
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        logger.debug(f"Response length: {len(response)} characters")
        return response.strip()
    
    def generate_vision_response(self, model_key: str, prompt: str, image_url: str = None, max_new_tokens: int = 150) -> str:
        """Generate response from a vision-language model"""
        logger.debug(f"Generating vision response for {model_key}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        logger.debug(f"Max new tokens: {max_new_tokens}")
        
        if model_key not in self.loaded_models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.loaded_models[model_key]
        processor = self.loaded_processors[model_key]
        
        # Check if this is a GGUF model
        if hasattr(model, 'create_completion'):  # llama-cpp-python model
            logger.debug(f"Using GGUF model for text generation")
            return self.generate_gguf_response(model, prompt, max_new_tokens)
        
        logger.debug(f"Using model device: {model.device}")
        
        # Load image
        if image_url is None:
            image_url = self.sample_image_url
        
        logger.debug(f"Loading image from: {image_url}")
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
            logger.debug(f"✓ Image loaded successfully, size: {image.size}")
        except Exception as e:
            logger.warning(f"❌ Failed to load image, using text-only fallback: {e}")
            # Fallback to text-only if image fails
            return self.generate_text_fallback(model_key, prompt, max_new_tokens)
        
        # Process inputs
        logger.debug(f"Processing text and image inputs for {model_key}...")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        logger.debug(f"Input keys: {list(inputs.keys())}")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.debug(f"Inputs moved to device: {model.device}")
        
        # Generate
        logger.debug(f"Starting vision-language generation for {model_key}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        logger.debug(f"Generation completed for {model_key}")
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Raw response length: {len(response)} characters")
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
            logger.debug(f"Cleaned response length: {len(response)} characters")
        
        return response
    
    def generate_gguf_response(self, model, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response using GGUF model (llama-cpp-python)"""
        logger.debug(f"Generating text with GGUF model, max tokens: {max_new_tokens}")
        
        try:
            response = model.create_completion(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                echo=False  # Don't include the prompt in response
            )
            
            generated_text = response['choices'][0]['text'].strip()
            logger.debug(f"GGUF response length: {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logger.error(f"GGUF generation failed: {e}")
            return f"[GGUF generation failed: {str(e)}]"
    
    def generate_text_fallback(self, model_key: str, prompt: str, max_new_tokens: int = 150) -> str:
        """Fallback text generation for vision models"""
        # This is a simplified fallback - in practice you'd need model-specific handling
        return f"[Text-only fallback for {model_key}] {prompt[:50]}... [Response would be generated here]"
    
    def test_model(self, model_key: str, model_info: Dict) -> List[BenchmarkResult]:
        """Test a single model with all appropriate prompts"""
        logger.info(f"🧪 Testing model: {model_key}")
        logger.debug(f"Model info: {model_info}")
        
        model_path = model_info["local_path"]
        model_type = model_info["model_type"]
        model_size = model_info.get("size_gb", "unknown")
        results = []
        
        logger.info(f"Model details - Path: {model_path}, Type: {model_type}, Size: {model_size} GB")
        
        try:
            # Load model based on type
            logger.info(f"Loading model {model_key} of type: {model_type}")
            if model_type == "vision-text-to-text":
                logger.debug(f"Loading as vision-language model...")
                self.load_vision_model(model_path, model_key)
                prompts = self.vision_prompts + self.test_prompts[:3]  # Test fewer prompts for vision models
                generation_func = self.generate_vision_response
                logger.info(f"✓ Vision model loaded. Will test {len(prompts)} prompts")
            else:
                logger.debug(f"Loading as text-only model...")
                self.load_text_model(model_path, model_key)
                prompts = self.test_prompts
                generation_func = self.generate_text_response
                logger.info(f"✓ Text model loaded. Will test {len(prompts)} prompts")
            
            # Test each prompt
            total_prompts = min(5, len(prompts))
            logger.info(f"🔄 Starting {total_prompts} prompt tests for {model_key}")
            
            for i, prompt in enumerate(prompts[:5]):  # Limit to 5 prompts per model to save time
                logger.info(f"📝 Testing prompt {i+1}/{total_prompts}: {prompt[:50]}...")
                logger.debug(f"Full prompt: {prompt}")
                
                try:
                    # Create inference function
                    def inference_func(p):
                        logger.debug(f"Calling generation function for {model_key}")
                        return generation_func(model_key, p)
                    
                    logger.debug(f"Starting benchmark for prompt {i+1}")
                    # Benchmark the inference
                    result = self.benchmark_runner.benchmark_inference(
                        model_name=f"{model_key}_prompt_{i+1}",
                        inference_function=inference_func,
                        prompt=prompt
                    )
                    
                    results.append(result)
                    
                    # Log result summary
                    if result.error:
                        logger.error(f"❌ Prompt {i+1} failed: {result.error}")
                    else:
                        logger.info(f"✅ Prompt {i+1} completed: {result.response_time_seconds:.2f}s, "
                                  f"{result.tokens_per_second:.1f} tokens/s, "
                                  f"memory: {result.peak_memory_gb:.1f}GB")
                        logger.debug(f"Response preview: {result.response[:100]}...")
                
                except Exception as e:
                    logger.error(f"❌ Failed to test prompt {i+1}: {e}")
                    logger.debug(f"Exception details: {type(e).__name__}: {e}")
                    # Create error result
                    error_result = BenchmarkResult(
                        model_name=f"{model_key}_prompt_{i+1}",
                        prompt=prompt,
                        response="",
                        response_time_seconds=0,
                        tokens_generated=0,
                        tokens_per_second=0,
                        peak_memory_gb=0,
                        avg_memory_gb=0,
                        peak_gpu_memory_gb=0,
                        avg_gpu_memory_gb=0,
                        avg_gpu_utilization=0,
                        max_gpu_temperature=0,
                        resource_snapshots=[],
                        error=str(e)
                    )
                    results.append(error_result)
                
                # Clean up GPU memory between prompts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        except Exception as e:
            logger.error(f"Failed to test model {model_key}: {e}")
            # Create error result for the entire model
            error_result = BenchmarkResult(
                model_name=f"{model_key}_failed",
                prompt="Model loading failed",
                response="",
                response_time_seconds=0,
                tokens_generated=0,
                tokens_per_second=0,
                peak_memory_gb=0,
                avg_memory_gb=0,
                peak_gpu_memory_gb=0,
                avg_gpu_memory_gb=0,
                avg_gpu_utilization=0,
                max_gpu_temperature=0,
                resource_snapshots=[],
                error=str(e)
            )
            results.append(error_result)
        
        finally:
            # Clean up loaded model
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
            if model_key in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_key]
            if model_key in self.loaded_processors:
                del self.loaded_processors[model_key]
            
            # Force cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return results
    
    def test_all_models(self, download_results_file: str) -> List[BenchmarkResult]:
        """Test all downloaded models"""
        # Load download results
        with open(download_results_file, 'r') as f:
            download_results = json.load(f)
        
        all_results = []
        
        for model_key, model_info in download_results.items():
            if "error" in model_info:
                logger.warning(f"Skipping {model_key} due to download error: {model_info['error']}")
                continue
            
            logger.info(f"Starting tests for {model_key}")
            model_results = self.test_model(model_key, model_info)
            all_results.extend(model_results)
            
            logger.info(f"Completed tests for {model_key}: {len(model_results)} results")
        
        return all_results
    
    def save_test_results(self, results: List[BenchmarkResult], output_file: str):
        """Save test results to file"""
        self.benchmark_runner.results = results
        self.benchmark_runner.save_results(output_file)

def main():
    """Main function to run model tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hugging Face models")
    parser.add_argument("--models-dir", default="./models", help="Directory with downloaded models")
    parser.add_argument("--download-results", default="./models/download_results.json", 
                       help="JSON file with download results")
    parser.add_argument("--output", default="./benchmark_results.json", 
                       help="Output file for benchmark results")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if download results exist
    if not os.path.exists(args.download_results):
        logger.error(f"Download results file not found: {args.download_results}")
        logger.info("Please run model_downloader.py first")
        return
    
    # Initialize tester
    tester = ModelTester(models_dir=args.models_dir, device=args.device)
    
    logger.info(f"Using device: {tester.device}")
    logger.info("Starting model testing...")
    
    # Run tests
    results = tester.test_all_models(args.download_results)
    
    # Save results
    tester.save_test_results(results, args.output)
    
    # Print summary
    summary = tester.benchmark_runner.get_summary_stats()
    logger.info("\n=== Test Summary ===")
    for key, value in summary.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()