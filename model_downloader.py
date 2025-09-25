#!/usr/bin/env python3
"""
Model Downloader and Configuration Manager
Downloads AI models from HuggingFace and prepares them for testing
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from huggingface_hub import hf_hub_download, HfApi, list_repo_files

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Downloads and manages AI model files from HuggingFace"""
    
    def __init__(self, models_dir: str = "./models"):
        logger.debug("🔧 Initializing ModelDownloader")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        logger.debug(f"📁 Models directory set to: {self.models_dir}")
        
        # Core model configurations - Llama-3.2-1B and TinyLlama as requested
        self.model_configs = {
            "llama-3.2-1b": {
                "hf_repo": "unsloth/Llama-3.2-1B-Instruct",  # Public Llama-3.2-1B model
                "type": "text-generation",
                "quantized_alternatives": [],  # Standard Transformers model
                "size_category": "small",
                "use_quantized": False,  # Use standard model
                "estimated_size_gb": 1.2,  # ~1.2GB as requested
                "prefer_q2k": False  # Standard model, no GGUF
            },
            "tinyllama": {
                "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Original TinyLlama model
                "type": "text-generation",
                "quantized_alternatives": [],  # Standard Transformers model
                "size_category": "tiny",
                "use_quantized": False,  # Use standard model
                "estimated_size_gb": 0.3,  # ~300MB as requested
                "prefer_q2k": False  # Standard model, no GGUF
            }
        }
        logger.debug(f"🔍 Model configurations loaded: {list(self.model_configs.keys())}")
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> Dict[str, Any]:
        """Download a model and return status information"""
        logger.debug(f"⬇️ Starting download process for model: {model_name}")
        
        if model_name not in self.model_configs:
            logger.debug(f"❌ Model '{model_name}' not found in configurations")
            return {
                "status": "error", 
                "error": f"Unknown model: {model_name}",
                "available_models": list(self.model_configs.keys())
            }
        
        config = self.model_configs[model_name]
        logger.debug(f"📋 Using configuration: {config}")
        
        try:
            # Check for quantized alternatives first
            if config.get("use_quantized", False):
                logger.debug("🔍 Checking for quantized alternatives")
                quantized_path = self._try_download_quantized(config, force_redownload)
                if quantized_path:
                    logger.debug(f"✅ Successfully downloaded quantized model: {quantized_path}")
                    return {
                        "status": "success",
                        "model": model_name,
                        "path": str(quantized_path),
                        "type": "quantized",
                        "files": [quantized_path.name]
                    }
                else:
                    logger.debug("⚠️ No quantized alternatives found, falling back to standard model")
            
            # Fall back to standard HuggingFace model
            logger.debug("📥 Downloading standard HuggingFace model")
            return self._download_hf_model(model_name, config, force_redownload)
            
        except Exception as e:
            logger.debug(f"💥 Download failed with error: {str(e)}")
            logger.error(f"Failed to download {model_name}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _try_download_quantized(self, config: Dict[str, Any], force_redownload: bool = False) -> Optional[Path]:
        """Try to download quantized alternatives"""
        logger.debug("🔍 Searching for quantized model files")
        
        repo_id = config["hf_repo"]
        logger.debug(f"📂 Checking repository: {repo_id}")
        
        try:
            # List all files in the repository (without authentication for public repos)
            api = HfApi(token=None)  # Explicitly use no token for public repos
            repo_files = list_repo_files(repo_id=repo_id, token=None)  # Pass token=None explicitly
            logger.debug(f"📋 Found {len(repo_files)} files in repository")
            
            # Filter for GGUF files
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            logger.debug(f"🔍 Found {len(gguf_files)} GGUF files: {gguf_files}")
            
            if not gguf_files:
                logger.debug("❌ No GGUF files found in repository")
                return None
            
            # Check for Q2_K preference first if enabled
            if config.get("prefer_q2k", False):
                logger.debug("🎯 Looking for Q2_K quantized files (preferred)")
                q2k_files = [f for f in gguf_files if "Q2_K" in f or "q2_k" in f.lower()]
                if q2k_files:
                    target_file = q2k_files[0]
                    logger.debug(f"✅ Found preferred Q2_K file: {target_file}")
                    return self._download_gguf_file(repo_id, target_file, force_redownload)
            
            # Try configured quantized alternatives
            for alt_name in config.get("quantized_alternatives", []):
                logger.debug(f"🔍 Checking for alternative: {alt_name}")
                if alt_name in gguf_files:
                    logger.debug(f"✅ Found matching file: {alt_name}")
                    return self._download_gguf_file(repo_id, alt_name, force_redownload)
            
            # If no specific alternatives, take the first available GGUF
            if gguf_files:
                target_file = gguf_files[0]
                logger.debug(f"📥 Using first available GGUF file: {target_file}")
                return self._download_gguf_file(repo_id, target_file, force_redownload)
            
        except Exception as e:
            logger.debug(f"💥 Quantized download attempt failed: {str(e)}")
            logger.error(f"Failed to download quantized model: {str(e)}")
        
        return None
    
    def _download_gguf_file(self, repo_id: str, filename: str, force_redownload: bool = False) -> Optional[Path]:
        """Download a specific GGUF file"""
        logger.debug(f"⬇️ Downloading GGUF file: {filename} from {repo_id}")
        
        local_path = self.models_dir / filename
        
        if local_path.exists() and not force_redownload:
            logger.debug(f"✅ File already exists locally: {local_path}")
            return local_path
        
        try:
            logger.debug("📥 Initiating HuggingFace download...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
                force_download=force_redownload,
                token=None  # Don't use authentication for public repos
            )
            logger.debug(f"✅ Successfully downloaded to: {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.debug(f"💥 GGUF download failed: {str(e)}")
            logger.error(f"Failed to download {filename}: {str(e)}")
            return None
    
    def _download_hf_model(self, model_name: str, config: Dict[str, Any], force_redownload: bool = False) -> Dict[str, Any]:
        """Download standard HuggingFace model"""
        logger.debug(f"📥 Downloading standard HuggingFace model: {model_name}")
        
        repo_id = config["hf_repo"]
        model_path = self.models_dir / model_name
        
        # Check if model already exists
        if model_path.exists() and not force_redownload:
            # Check if the directory has files
            files = list(model_path.glob("*.json")) + list(model_path.glob("*.bin")) + list(model_path.glob("*.safetensors"))
            if files:
                logger.debug(f"✅ Model {model_name} already exists with {len(files)} files")
                return {
                    "status": "success",
                    "message": "Model already downloaded", 
                    "model": model_name,
                    "path": str(model_path),
                    "files": [f.name for f in files[:5]]  # Show first 5 files
                }
        
        model_path.mkdir(exist_ok=True)
        logger.debug(f"📂 Model will be saved to: {model_path}")
        
        try:
            from huggingface_hub import snapshot_download
            
            logger.debug(f"⬇️ Starting download of {repo_id}...")
            logger.info(f"Downloading {model_name} from {repo_id}...")
            
            # Download the complete model without authentication
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=None,  # Don't use authentication for public repos
                allow_patterns=None,  # Download all files
                ignore_patterns=None
            )
            
            # Count downloaded files
            files = list(model_path.glob("*.*"))
            logger.debug(f"✅ Downloaded {len(files)} files to {model_path}")
            logger.info(f"✅ Successfully downloaded {model_name}")
            
            return {
                "status": "success",
                "message": f"Successfully downloaded {len(files)} files", 
                "model": model_name,
                "path": str(model_path),
                "files": [f.name for f in files[:5]]  # Show first 5 files
            }
            
        except Exception as e:
            logger.debug(f"💥 Standard model download failed: {str(e)}")
            logger.error(f"Failed to download {model_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model": model_name,
                "path": str(model_path)
            }
    
    def list_available_models(self) -> List[str]:
        """Return list of available model configurations"""
        logger.debug("📋 Listing available model configurations")
        models = list(self.model_configs.keys())
        logger.debug(f"✅ Found {len(models)} configured models: {models}")
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        logger.debug(f"ℹ️ Getting information for model: {model_name}")
        
        if model_name not in self.model_configs:
            logger.debug(f"❌ Model '{model_name}' not found")
            return None
        
        info = self.model_configs[model_name].copy()
        logger.debug(f"✅ Retrieved model info: {info}")
        return info
    
    def check_downloaded_models(self) -> Dict[str, Any]:
        """Check which models are already downloaded"""
        logger.debug("🔍 Checking for already downloaded models")
        
        downloaded = {}
        
        for model_name in self.model_configs.keys():
            logger.debug(f"🔍 Checking model: {model_name}")
            model_files = []
            
            # Check for GGUF files
            gguf_files = list(self.models_dir.glob("*.gguf"))
            logger.debug(f"📁 Found {len(gguf_files)} GGUF files in models directory")
            
            for gguf_file in gguf_files:
                if model_name.replace("-", "_") in gguf_file.name.lower():
                    model_files.append(str(gguf_file))
                    logger.debug(f"✅ Found matching file: {gguf_file.name}")
            
            # Check for model directories
            model_dir = self.models_dir / model_name
            if model_dir.exists() and model_dir.is_dir():
                logger.debug(f"📂 Found model directory: {model_dir}")
                model_files.append(str(model_dir))
            
            if model_files:
                downloaded[model_name] = {
                    "files": model_files,
                    "status": "available"
                }
                logger.debug(f"✅ Model {model_name} is available locally")
            else:
                logger.debug(f"❌ Model {model_name} not found locally")
        
        logger.debug(f"📊 Download check complete. Available models: {list(downloaded.keys())}")
        return downloaded
    
    def download_all_models(self, force_download: bool = False) -> Dict[str, Any]:
        """Download all configured models"""
        logger.debug("🚀 Starting download of all configured models")
        
        results = {}
        
        for model_name in self.model_configs.keys():
            logger.debug(f"📥 Processing model: {model_name}")
            
            try:
                result = self.download_model(model_name, force_download)
                
                # Convert result to expected format for orchestrator
                if result["status"] == "success":
                    # Use configured size from model config
                    estimated_size = self.model_configs[model_name].get("estimated_size_gb", 1.0)
                    
                    results[model_name] = {
                        "repo_id": self.model_configs[model_name]["hf_repo"],
                        "size_gb": estimated_size,
                        "files": result.get("files", []),
                        "path": result["path"],
                        "type": result.get("type", "quantized")
                    }
                    logger.debug(f"✅ {model_name} download successful")
                    
                elif result["status"] == "partial":
                    # Handle partial success (standard HF model not fully implemented)
                    # Use configured size from model config
                    estimated_size = self.model_configs[model_name].get("estimated_size_gb", 1.0)
                    # Get model type from configuration
                    model_type = self.model_configs[model_name].get("type", "text-generation")
                    results[model_name] = {
                        "repo_id": self.model_configs[model_name]["hf_repo"],
                        "size_gb": estimated_size,
                        "files": [],
                        "path": result["path"],
                        "type": model_type,
                        "status": "partial",
                        "message": result["message"]
                    }
                    logger.debug(f"⚠️ {model_name} partially successful: {result['message']}")
                    
                else:
                    # Handle errors
                    results[model_name] = {
                        "error": result["error"]
                    }
                    logger.debug(f"❌ {model_name} download failed: {result['error']}")
                    
            except Exception as e:
                logger.debug(f"💥 Exception downloading {model_name}: {str(e)}")
                results[model_name] = {
                    "error": str(e)
                }
        
        # Save results to file for orchestrator
        results_file = self.models_dir / "download_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.debug(f"📊 Download results saved to {results_file}")
        logger.debug(f"🎯 Download all models complete. Results: {len(results)} models processed")
        
        return results

def main():
    """Main function for testing model downloader"""
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 Starting Model Downloader test")
    
    downloader = ModelDownloader()
    
    # List available models
    logger.info("📋 Available model configurations:")
    for model in downloader.list_available_models():
        info = downloader.get_model_info(model)
        logger.info(f"  - {model}: {info}")
    
    # Check already downloaded
    logger.info("🔍 Checking downloaded models:")
    downloaded = downloader.check_downloaded_models()
    for model, info in downloaded.items():
        logger.info(f"  - {model}: {info}")
    
    # Test download
    logger.info("⬇️ Testing download:")
    result = downloader.download_model("tinyllama")
    logger.info(f"Download result: {result}")

if __name__ == "__main__":
    main()