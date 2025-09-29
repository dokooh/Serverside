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
        
        # Comprehensive prompt sets - 10 prompts total covering all 7 types
        # Each model will be tested with all these prompts
        self.comprehensive_prompts = [
            # Web Search Tool prompts (2 prompts)
            "Get the public information about Tom Brady, use web-search tool.",
            "Search for the latest stock price of Tesla using web-search tool.",
            
            # Calculation Tool prompts (2 prompts) 
            "Calculate 30% of 12000.",
            "What is 25 * 48 + 137?",
            
            # Image Analysis Tool prompts (2 prompts)
            "User uploaded image, caption the purchase order details.",
            "User uploaded image, extract and list all product information.",
            
            # OCR prompts (1 prompt)
            "Extract all visible text from this image and format it properly.",
            
            # Document Analysis prompts (1 prompt)
            "What type of document is this? (invoice, receipt, form, letter, etc.)",
            
            # Image QA prompts (1 prompt)
            "How many people are in this image?",
            
            # Vision Description prompts (1 prompt)
            "Describe what you see in this image."
        ]
        
        # Legacy tool selection prompts (kept for backward compatibility)
        self.tool_selection_prompts = [
            "Get the public information about Tom Brady, use web-search tool.",
            "Search for the latest stock price of Tesla using web-search tool.",
            "Find information about climate change effects using web-search tool.",
            "Look up the definition of quantum computing using web-search tool.",
            "Research the current population of Japan using web-search tool.",
            "Calculate 30% of 12000.",
            "What is 25 * 48 + 137?",
            "Compute the square root of 2025.",
            "Calculate the compound interest on $5000 at 3% for 5 years.",
            "Find the area of a circle with radius 7.5."
        ]
        
        # Vision prompts for vision-capable models
        self.vision_prompts = [
            "Describe what you see in this image.",
            "What objects are present in this image?",
            "What is the main subject of this image?"
        ]
        
        # Model-specific system prompts optimized for JSON tool calling
        self.system_prompts = {
            "llama-3.2-1b": """You are an assistant that MUST respond exactly with one valid JSON object and nothing else.
Tools:
- transcribe_image(image_id:string, fields:array[string])
- web_search(query:string)
- calculate(expression:string)

When given an image instruction, prefer transcribe_image. When asked to look something up, use web_search.
Respond with valid JSON only.""",
            
            "tinyllama": """You are an assistant that MUST respond with a single valid JSON object and nothing else.
Available tools (pick exactly one):
1) transcribe_image(image_id:string, fields:array[string])
2) web_search(query:string)
3) calculate(expression:string)

EXAMPLES:
Input: "Image img_001: Please transcribe the invoice and return fields date, vendor, total."
Output: {"name":"transcribe_image","arguments":{"image_id":"img_001","fields":["date","vendor","total"]}}

Input: "Find the release date for 'ExampleSoft Editor'. Use web_search."
Output: {"name":"web_search","arguments":{"query":"release date ExampleSoft Editor"}}

Input: "Calculate 100 * 0.15 and return as calculate call."
Output: {"name":"calculate","arguments":{"expression":"100 * 0.15"}}
END EXAMPLES
Respond only with the JSON object for the selected tool."""
        }
        
        # Model-specific optimized test prompts
        self.model_specific_prompts = {
            "llama-3.2-1b": [
                # Basic transcription
                "Image inv_A12: Transcribe fields date, vendor, total, invoice_no.",
                
                # Web lookup with refinement
                "Find the latest stable release of ExampleSoft Editor and return as web_search.",
                
                # Ambiguous screenshot — rule enforcement
                "Screenshot attached. If it contains >5 words of visible instructions use transcribe_image; otherwise use web_search.",
                
                # Chained intent (choose first tool)
                "I uploaded a screenshot of a product label and asked what it is and its manufacturer. Decide the first tool to call (transcribe_image or web_search) and return that single call.",
                
                # Numeric compute
                "Please compute monthly payment for 50000 at 3.5% annual over 60 months. Return calculate(expression).",
                
                # Robustness test — truncated prompt
                "Image x9: Transcribe date, vendor",
                
                # Additional web search test
                "Look up the current stock price of NVIDIA Corporation using web_search.",
                
                # Additional transcription test
                "Image doc_contract_99: Extract fields client_name, contract_date, total_amount.",
                
                # Additional calculation test
                "Calculate the sales tax on $2500 at 8.5% rate using calculate tool.",
                
                # Additional rule enforcement test
                "Image scan_receipt_33: If document type is receipt, transcribe fields store, date, total; otherwise use web_search."
            ],
            
            "tinyllama": [
                # Simple transcription
                "Image img_001: Please transcribe the invoice and return fields date, vendor, total.",
                
                # Simple web lookup
                "Please use web_search to find the release date for \"ExampleSoft Editor\".",
                
                # Ambiguous — decide tool (with binary rule)
                "User uploaded a screenshot and asked \"what should I do next?\" If >50% text visible, transcribe; otherwise web_search for product info.",
                
                # Short calculation
                "Calculate the VAT on 1234.50 at 20% and return as a calculate call.",
                
                # Multiple fields (receipt)
                "Image img_010: Transcribe receipt and return fields vendor, date, total, tax.",
                
                # Malformed-output stress test
                "Image img_020: Transcribe and return date only.",
                
                # Simple web search
                "Use web_search to find \"Python tutorial basics\".",
                
                # Basic math
                "Calculate 150 + 75 * 2 using the calculate tool.",
                
                # Simple image task
                "Image img_030: Get field price only.",
                
                # Direct tool call
                "Search web for \"weather today\" using web_search tool."
            ]
        }
        
        # Extended prompt categories for detailed testing
        # Web Search Tool prompts
        self.web_search_prompts = [
            "Get the public information about Tom Brady, use web-search tool.",
            "Search for the latest stock price of Tesla using web-search tool.",
            "Find information about climate change effects using web-search tool.",
            "Look up the definition of quantum computing using web-search tool.",
            "Research the current population of Japan using web-search tool."
        ]
        
        # Calculation Tool prompts
        self.calculation_prompts = [
            "Calculate 30% of 12000.",
            "What is 25 * 48 + 137?",
            "Compute the square root of 2025.",
            "Calculate the compound interest on $5000 at 3% for 5 years.",
            "Find the area of a circle with radius 7.5."
        ]
        
        # Image analysis tool selection prompts - for models to recognize they need image analysis tools
        self.image_analysis_prompts = [
            "User uploaded image, caption the purchase order details.",
            "User uploaded image, extract and list all product information.",
            "User uploaded image, identify the document type and summarize contents.",
            "User uploaded image, extract all monetary amounts and totals.",
            "User uploaded image, caption the business card details."
        ]
        
        # Document processing prompts for OCR and document understanding
        self.document_prompts = [
            "Extract all visible text from this image and format it properly.",
            "What type of document is this? (invoice, receipt, form, letter, etc.)",
            "List all numbers, dates, and important information visible in this document.",
            "Summarize the key points from this document image.",
            "What is the main purpose or content of this document?",
            "Identify any tables, forms, or structured data in this image.",
            "Extract contact information (phone numbers, emails, addresses) if visible.",
            "What language is this document written in?",
            "Are there any signatures, logos, or special markings in this document?",
            "Convert this handwritten text to digital text."
        ]
        
        # Image QA prompts for detailed visual analysis
        self.image_qa_prompts = [
            "How many people are in this image?",
            "What colors are most prominent in this image?",
            "What time of day does this appear to be taken?",
            "Is this an indoor or outdoor scene?",
            "What emotions or mood does this image convey?",
            "What brand names or text can you see in this image?",
            "Describe the lighting conditions in this image.",
            "What activities are happening in this image?",
            "What is the approximate age or era of this image?",
            "Are there any safety concerns visible in this image?"
        ]
        
        # Sample image URL for vision models
        self.sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        
        # Sample document images for OCR and document processing tests
        self.document_images = [
            # Reliable document samples from different sources
            "https://via.placeholder.com/800x600/ffffff/000000?text=SAMPLE+INVOICE%0A%0AInvoice+%23001%0ADate%3A+2024-01-01%0A%0AFrom%3A+Company+ABC%0ATo%3A+Customer+XYZ%0A%0AItems%3A%0A1.+Service+A+-+%24100%0A2.+Product+B+-+%24200%0A%0ATotal%3A+%24300",
            "https://via.placeholder.com/800x600/f0f0f0/333333?text=BUSINESS+LETTER%0A%0AJanuary+1%2C+2024%0A%0ADear+Customer%2C%0A%0AThank+you+for+your+business.%0AWe+appreciate+your+support.%0A%0ASincerely%2C%0AManagement+Team",
            "https://via.placeholder.com/600x800/ffffff/000000?text=FORM+SAMPLE%0A%0AName%3A+________________%0A%0AEmail%3A+________________%0A%0APhone%3A+________________%0A%0AAddress%3A%0A__________________________%0A__________________________%0A%0ASignature%3A+____________",
            # Fallback to the main sample image if document images fail
            self.sample_image_url
        ]
        
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.loaded_processors = {}
    
    def get_system_prompt(self, model_key: str) -> str:
        """Get optimized system prompt for specific model"""
        model_key_lower = model_key.lower()
        if "llama-3.2" in model_key_lower or "llama3.2" in model_key_lower:
            return self.system_prompts["llama-3.2-1b"]
        elif "tinyllama" in model_key_lower:
            return self.system_prompts["tinyllama"]
        else:
            return "You are a helpful assistant that uses tools to complete tasks. Respond with valid JSON for tool calls."
    
    def get_model_prompts(self, model_key: str) -> List[str]:
        """Get optimized prompts for specific model"""
        model_key_lower = model_key.lower()
        if "llama-3.2" in model_key_lower or "llama3.2" in model_key_lower:
            return self.model_specific_prompts["llama-3.2-1b"]
        elif "tinyllama" in model_key_lower:
            return self.model_specific_prompts["tinyllama"]
        else:
            # Fallback to comprehensive prompts for other models
            return self.comprehensive_prompts
    
    def get_quantization_config(self) -> BitsAndBytesConfig:
        """Get 4-bit quantization config for memory efficiency"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    def categorize_prompt(self, prompt: str) -> str:
        """Categorize prompt type for better testing context"""
        prompt_lower = prompt.lower()
        
        # JSON tool-calling prompts (new optimized prompts)
        if any(keyword in prompt_lower for keyword in ['transcribe_image', 'image', 'transcribe']):
            return "🖼️ Image Transcription Tool"
        elif any(keyword in prompt_lower for keyword in ['web_search', 'web-search', 'search for', 'find', 'lookup']):
            return "🔍 Web Search Tool"
        elif any(keyword in prompt_lower for keyword in ['calculate', 'compute', 'vat', 'payment', '*', '+', '-', '/', '%']):
            return "🧮 Calculation Tool"
        
        # Legacy prompt categories
        elif any(keyword in prompt_lower for keyword in ['user uploaded image', 'caption the purchase', 'extract and list all product']):
            return "🖼️ Image Analysis Tool"
        elif any(keyword in prompt_lower for keyword in ['extract all visible text', 'visible text', 'format it properly']):
            return "📄 OCR"
        elif any(keyword in prompt_lower for keyword in ['what type of document', 'invoice', 'receipt', 'form', 'letter']):
            return "📋 Document Analysis"
        elif any(keyword in prompt_lower for keyword in ['how many people', 'count', 'colors', 'time of day', 'indoor', 'outdoor']):
            return "❓ Image QA"
        elif any(keyword in prompt_lower for keyword in ['describe what you see', 'see in this image', 'objects', 'main subject']):
            return "👁️ Vision Description"
        else:
            return "� Tool Selection"
    
    def is_gguf_model(self, model_path: str) -> bool:
        """Check if model directory contains GGUF files"""
        model_dir = Path(model_path)
        return any(f.suffix.lower() == '.gguf' for f in model_dir.rglob('*.gguf'))
    
    def find_gguf_file(self, model_path: str, prefer_q2k_only: bool = True) -> Optional[str]:
        """Find Q2_K GGUF file in the model directory"""
        logger.debug(f"🔍 Debug - Searching for Q2_K GGUF files in {model_path}")
        
        model_dir = Path(model_path)
        gguf_files = list(model_dir.rglob('*.gguf'))
        logger.debug(f"📁 Debug - Found {len(gguf_files)} GGUF files: {[f.name for f in gguf_files]}")
        
        if not gguf_files:
            logger.debug(f"❌ Debug - No GGUF files found in {model_path}")
            return None
        
        # Look specifically for Q2_K quantization
        logger.debug(f"🔍 Debug - Looking for Q2_K variant only...")
        for gguf_file in gguf_files:
            if 'Q2_K' in gguf_file.name:
                logger.info(f"✅ Found Q2_K GGUF file: {gguf_file}")
                logger.debug(f"📏 Debug - File size: {gguf_file.stat().st_size / (1024**3):.2f} GB")
                return str(gguf_file)
        
        logger.warning(f"⚠️ No Q2_K variant found, falling back to first available GGUF file")
        
        # Fallback: return the first GGUF file found (should not happen if Q2_K is properly downloaded)
        selected_file = gguf_files[0]
        logger.warning(f"📄 Using first available GGUF file as fallback: {selected_file}")
        logger.debug(f"📏 Debug - File size: {selected_file.stat().st_size / (1024**3):.2f} GB")
        return str(selected_file)
    
    def load_gguf_model(self, model_path: str, model_key: str) -> Tuple[Any, Any]:
        """Load a GGUF model using llama-cpp-python"""
        logger.debug(f"🔧 Debug - Loading GGUF model: {model_key} from {model_path}")
        logger.debug(f"🔧 Debug - GGUF support available: {GGUF_AVAILABLE}")
        
        if not GGUF_AVAILABLE:
            logger.warning(f"⚠️ GGUF support not available for {model_key}, falling back to transformers")
            logger.debug(f"🔄 Debug - Calling fallback method for {model_key}")
            return self.load_vision_model_fallback(model_path, model_key)
        
        logger.info(f"🔧 Loading GGUF model: {model_key}")
        
        logger.debug(f"🔍 Debug - Finding GGUF file in {model_path}...")
        gguf_file = self.find_gguf_file(model_path)
        if not gguf_file:
            logger.debug(f"❌ Debug - No GGUF file found in {model_path}")
            raise FileNotFoundError(f"No GGUF file found in {model_path}")
        
        logger.debug(f"✅ Debug - Using GGUF file: {gguf_file}")
        
        try:
            # Load GGUF model with llama-cpp-python
            logger.info(f"📥 Loading GGUF file: {gguf_file}")
            logger.debug(f"🔧 Debug - Device: {self.device}")
            logger.debug(f"🔧 Debug - GPU layers: {-1 if self.device == 'cuda' else 0}")
            logger.debug(f"🔧 Debug - Context length: 2048")
            
            logger.debug(f"🚀 Debug - Initializing Llama model...")
            model = Llama(
                model_path=gguf_file,
                n_ctx=2048,  # Context length
                n_gpu_layers=-1 if self.device == "cuda" else 0,  # Use GPU if available
                verbose=False
            )
            logger.debug(f"✅ Debug - Llama model initialized successfully")
            
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
    
    def load_text_model(self, model_path: str, model_key: str, repo_id: str = None) -> Tuple[Any, Any]:
        """Load a text generation model"""
        logger.info(f"Loading text model: {model_key}")
        logger.debug(f"Model path: {model_path}")
        logger.debug(f"Device: {self.device}")
        
        # Check if this is a GGUF model
        if self.is_gguf_model(model_path):
            logger.info(f"Detected GGUF model for {model_key}")
            return self.load_gguf_model(model_path, model_key)
        
        try:
            # Try to load tokenizer
            logger.debug(f"Loading tokenizer for {model_key}...")
            
            # Use repo_id for tokenizer instead of local path if available
            tokenizer_path = repo_id if repo_id else model_path
            logger.debug(f"Using tokenizer path: {tokenizer_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
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
            
            # Configure FlashAttention2
            try:
                import flash_attn
                logger.debug(f"FlashAttention2 detected - enabling for {model_key}")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.debug(f"FlashAttention2 not available - using standard attention for {model_key}")
                model_kwargs["attn_implementation"] = "eager"
            
            if self.device == "cuda":
                logger.debug(f"Using CUDA configuration with quantization for {model_key}")
                model_kwargs["quantization_config"] = self.get_quantization_config()
                model_kwargs["device_map"] = "auto"
            else:
                logger.debug(f"Using CPU configuration for {model_key}")
            
            logger.debug(f"Model kwargs: {model_kwargs}")
            logger.info(f"Loading model weights for {model_key}... (this may take a while)")
            
            model = AutoModelForCausalLM.from_pretrained(
                tokenizer_path,  # Use repo_id instead of local path for model loading too
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
    
    def load_vision_model(self, model_path: str, model_key: str, repo_id: str = None) -> Tuple[Any, Any]:
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
            processor_repo = repo_id if repo_id else model_path
            processor = AutoProcessor.from_pretrained(
                processor_repo,
                trust_remote_code=True
            )
            logger.debug(f"✓ Processor loaded successfully for {model_key}")
            
            logger.debug(f"Preparing vision model configuration for {model_key}...")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            # Configure FlashAttention2 for vision model
            try:
                import flash_attn
                logger.debug(f"FlashAttention2 detected - enabling for vision model {model_key}")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.debug(f"FlashAttention2 not available - using standard attention for vision model {model_key}")
                model_kwargs["attn_implementation"] = "eager"
            
            if self.device == "cuda":
                logger.debug(f"Using CUDA configuration for vision model {model_key}")
                model_kwargs["device_map"] = "auto"
            else:
                logger.debug(f"Using CPU configuration for vision model {model_key}")
            
            logger.debug(f"Vision model kwargs: {model_kwargs}")
            logger.info(f"Loading vision model weights for {model_key}... (this may take a while)")
            
            # Use repo_id for vision model loading too
            model_repo = repo_id if repo_id else model_path
            model = AutoModelForCausalLM.from_pretrained(
                model_repo,
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
                temperature=0.1,  # Lower temperature for deterministic JSON output
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
        
        # Select appropriate image based on prompt type
        if image_url is None:
            if any(keyword in prompt.lower() for keyword in ['text', 'ocr', 'document', 'extract', 'handwritten', 'invoice', 'receipt', 'form', 'letter']):
                # Use document image for OCR/document processing prompts
                import random
                image_url = random.choice(self.document_images)
                logger.debug(f"🔍 Using document image for OCR/document prompt")
            else:
                # Use general image for other vision prompts
                image_url = self.sample_image_url
                logger.debug(f"🖼️ Using general image for vision prompt")
        
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
                temperature=0.1,  # Lower temperature for deterministic JSON output
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
                temperature=0.1,  # Lower temperature for deterministic JSON output
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
        logger.debug(f"🔧 Debug - Model info: {model_info}")
        logger.debug(f"🔧 Debug - Starting model test procedure...")
        
        model_path = model_info["path"]  # Changed from "local_path" to "path"
        model_type = model_info.get("type", "text-generation")  # Changed from "model_type" to "type" with default
        model_size = model_info.get("size_gb", "unknown")
        results = []
        
        logger.info(f"Model details - Path: {model_path}, Type: {model_type}, Size: {model_size} GB")
        
        try:
            # Load model based on type
            logger.info(f"Loading model {model_key} of type: {model_type}")
            if model_type == "vision-text-to-text":
                logger.debug(f"Loading as vision-language model...")
                repo_id = model_info.get('repo_id', model_key)
                self.load_vision_model(model_path, model_key, repo_id)
                generation_func = self.generate_vision_response
                logger.info(f"✓ Vision model loaded. Will test all 10 comprehensive prompts (all 7 types)")
            else:
                logger.debug(f"Loading as text-only model...")
                repo_id = model_info.get('repo_id', model_key)
                self.load_text_model(model_path, model_key, repo_id)
                generation_func = self.generate_text_response
                logger.info(f"✓ Text model loaded. Will test all 10 comprehensive prompts (all 7 types)")
            
            # Get model-specific optimized prompts and system prompt
            system_prompt = self.get_system_prompt(model_key)
            prompts = self.get_model_prompts(model_key)
            total_prompts = len(prompts)
            
            logger.info(f"🔄 Starting {total_prompts} optimized prompt tests for {model_key}")
            logger.info(f"📋 Using model-specific system prompt ({len(system_prompt)} chars)")
            logger.debug(f"System prompt preview: {system_prompt[:100]}...")
            
            for i, prompt in enumerate(prompts):  # Test all optimized prompts
                prompt_category = self.categorize_prompt(prompt)
                logger.info(f"📝 Testing prompt {i+1}/{total_prompts} [{prompt_category}]: {prompt[:50]}...")
                logger.debug(f"Full prompt: {prompt}")
                
                try:
                    # Create inference function with system prompt
                    def inference_func(p):
                        logger.debug(f"Calling generation function for {model_key} with system prompt")
                        # Format prompt with system prompt for better tool calling
                        formatted_prompt = f"System: {system_prompt}\n\nUser: {p}\n\nAssistant:"
                        return generation_func(model_key, formatted_prompt)
                    
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