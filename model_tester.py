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
    
    def load_text_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a text generation model"""
        logger.info(f"Loading text model: {model_key}")
        
        try:
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization if needed
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                model_kwargs["quantization_config"] = self.get_quantization_config()
                model_kwargs["device_map"] = "auto"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.loaded_models[model_key] = model
            self.loaded_tokenizers[model_key] = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load text model {model_key}: {e}")
            raise
    
    def load_vision_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a vision-language model"""
        logger.info(f"Loading vision model: {model_key}")
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.loaded_models[model_key] = model
            self.loaded_processors[model_key] = processor
            
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load vision model {model_key}: {e}")
            raise
    
    def generate_text_response(self, model_key: str, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate text response from a text model"""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.loaded_models[model_key]
        tokenizer = self.loaded_tokenizers[model_key]
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response (only the new tokens)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def generate_vision_response(self, model_key: str, prompt: str, image_url: str = None, max_new_tokens: int = 150) -> str:
        """Generate response from a vision-language model"""
        if model_key not in self.loaded_models:
            raise ValueError(f"Model {model_key} not loaded")
        
        model = self.loaded_models[model_key]
        processor = self.loaded_processors[model_key]
        
        # Load image
        if image_url is None:
            image_url = self.sample_image_url
        
        try:
            image = Image.open(requests.get(image_url, stream=True).raw)
        except Exception as e:
            logger.warning(f"Failed to load image, using text-only: {e}")
            # Fallback to text-only if image fails
            return self.generate_text_fallback(model_key, prompt, max_new_tokens)
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        return response
    
    def generate_text_fallback(self, model_key: str, prompt: str, max_new_tokens: int = 150) -> str:
        """Fallback text generation for vision models"""
        # This is a simplified fallback - in practice you'd need model-specific handling
        return f"[Text-only fallback for {model_key}] {prompt[:50]}... [Response would be generated here]"
    
    def test_model(self, model_key: str, model_info: Dict) -> List[BenchmarkResult]:
        """Test a single model with all appropriate prompts"""
        logger.info(f"Testing model: {model_key}")
        
        model_path = model_info["local_path"]
        model_type = model_info["model_type"]
        results = []
        
        try:
            # Load model based on type
            if model_type == "vision-text-to-text":
                self.load_vision_model(model_path, model_key)
                prompts = self.vision_prompts + self.test_prompts[:3]  # Test fewer prompts for vision models
                generation_func = self.generate_vision_response
            else:
                self.load_text_model(model_path, model_key)
                prompts = self.test_prompts
                generation_func = self.generate_text_response
            
            # Test each prompt
            for i, prompt in enumerate(prompts[:5]):  # Limit to 5 prompts per model to save time
                logger.info(f"Testing prompt {i+1}/{min(5, len(prompts))}: {prompt[:50]}...")
                
                try:
                    # Create inference function
                    def inference_func(p):
                        return generation_func(model_key, p)
                    
                    # Benchmark the inference
                    result = self.benchmark_runner.benchmark_inference(
                        model_name=f"{model_key}_prompt_{i+1}",
                        inference_function=inference_func,
                        prompt=prompt
                    )
                    
                    results.append(result)
                    
                    # Log result summary
                    if result.error:
                        logger.error(f"Prompt {i+1} failed: {result.error}")
                    else:
                        logger.info(f"Prompt {i+1} completed: {result.response_time_seconds:.2f}s, "
                                  f"{result.tokens_per_second:.1f} tokens/s")
                
                except Exception as e:
                    logger.error(f"Failed to test prompt {i+1}: {e}")
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