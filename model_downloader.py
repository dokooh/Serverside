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
        
        # Core model configurations - updated for Vicuna-7B
        self.model_configs = {
            "vicuna-7b-v1.5": {
                "hf_repo": "lmsys/vicuna-7b-v1.5",
                "quantized_alternatives": [
                    "vicuna-7b-v1.5.Q2_K.gguf",  # Requested Q2_K preference
                    "vicuna-7b-v1.5.Q4_K_M.gguf", 
                    "vicuna-7b-v1.5.q4_0.gguf"
                ],
                "size_category": "medium",
                "use_quantized": True,
                "prefer_q2k": True  # Special preference for Q2_K as requested
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
            # List all files in the repository
            api = HfApi()
            repo_files = list_repo_files(repo_id=repo_id)
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
                force_download=force_redownload
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
        model_path.mkdir(exist_ok=True)
        
        logger.debug(f"📂 Model will be saved to: {model_path}")
        
        try:
            # This is a simplified approach - in practice you might use:
            # snapshot_download(repo_id, local_dir=model_path)
            logger.debug("⚠️ Standard HuggingFace download not fully implemented")
            return {
                "status": "partial",
                "message": "Standard model download needs implementation", 
                "model": model_name,
                "path": str(model_path)
            }
            
        except Exception as e:
            logger.debug(f"💥 Standard model download failed: {str(e)}")
            raise
    
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
    result = downloader.download_model("vicuna-7b-v1.5")
    logger.info(f"Download result: {result}")

if __name__ == "__main__":
    main()