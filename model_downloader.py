#!/usr/bin/env python3
"""
Model Downloader Script for Hugging Face Models
Downloads the smallest/quantized versions of specified models for Kaggle environment
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import snapshot_download, login, HfApi
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handles downloading of Hugging Face models with preference for quantized versions"""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.api = HfApi()
        
        # Model configurations with preferred quantized versions
        self.models_config = {
            "llama-3.2-1b": {
                "primary": "meta-llama/Llama-3.2-1B",
                "quantized_alternatives": [
                    "unsloth/Llama-3.2-1B-bnb-4bit",
                    "microsoft/Llama-3.2-1B-Instruct-GGUF",
                    "bartowski/Llama-3.2-1B-GGUF"
                ],
                "type": "text-generation"
            },
            "tinyllama": {
                "primary": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "quantized_alternatives": [
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF",
                    "microsoft/TinyLlama-1.1B-Chat-v1.0-onnx"
                ],
                "type": "text-generation"
            },
            "smolvlm-instruct": {
                "primary": "HuggingFaceTB/SmolVLM-Instruct",
                "quantized_alternatives": [
                    "bartowski/SmolVLM-Instruct-GGUF",
                    "mradermacher/SmolVLM-Instruct-GGUF"
                ],
                "type": "vision-text-to-text"
            }
        }
    
    def get_model_size(self, repo_id: str) -> Optional[float]:
        """Get approximate model size in GB"""
        try:
            repo_info = self.api.repo_info(repo_id)
            total_size = 0
            
            for sibling in repo_info.siblings:
                if sibling.size:
                    total_size += sibling.size
            
            return total_size / (1024**3)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get size for {repo_id}: {e}")
            return None
    
    def find_best_model_variant(self, model_key: str) -> str:
        """Find the smallest available model variant"""
        config = self.models_config[model_key]
        
        # For SmolVLM, target specifically the 1.1GB Q4_K_M GGUF
        if model_key == "smolvlm-instruct":
            for alt_repo in config["quantized_alternatives"]:
                try:
                    # Check if this is a GGUF repo and has the right size Q4_K_M variant
                    if "GGUF" in alt_repo:
                        repo_info = self.api.repo_info(alt_repo)
                        # Look specifically for Q4_K_M files around 1.1GB
                        for sibling in repo_info.siblings:
                            if "Q4_K_M" in sibling.rfilename and sibling.size:
                                file_size_gb = sibling.size / (1024**3)
                                # Target files around 1.1GB (allow 0.9-1.3GB range)
                                if 0.9 <= file_size_gb <= 1.3:
                                    logger.info(f"Found target 1.1GB Q4_K_M GGUF file: {sibling.rfilename} ({file_size_gb:.2f} GB)")
                                    return alt_repo
                                else:
                                    logger.debug(f"Skipping Q4_K_M file {sibling.rfilename} ({file_size_gb:.2f} GB) - not target size")
                except Exception as e:
                    logger.warning(f"Could not check Q4_K_M files in {alt_repo}: {e}")
                    continue
            
            # If no suitable GGUF found, check primary model size
            primary = config["primary"]
            size = self.get_model_size(primary)
            if size and 0.9 <= size <= 1.3:
                logger.info(f"Using primary model {primary} ({size:.2f} GB) - matches target size")
                return primary
            else:
                logger.warning(f"Primary model {primary} ({size:.2f} GB if available) doesn't match target 1.1GB size")
                return primary  # Use anyway as fallback
        
        # For other models, use standard logic
        smallest_size = float('inf')
        best_variant = None
        
        for alt_repo in config["quantized_alternatives"]:
            try:
                size = self.get_model_size(alt_repo)
                if size is not None and size < smallest_size:
                    smallest_size = size
                    best_variant = alt_repo
                    logger.info(f"Found quantized variant {alt_repo} ({size:.2f} GB)")
            except Exception as e:
                logger.warning(f"Could not access {alt_repo}: {e}")
                continue
        
        if best_variant:
            return best_variant
        
        # Fall back to primary model
        primary = config["primary"]
        size = self.get_model_size(primary)
        logger.info(f"Using primary model {primary} ({size:.2f} GB if available)")
        return primary
    
    def download_model(self, model_key: str, force_download: bool = False) -> Dict:
        """Download a model and return metadata"""
        logger.info(f"Starting download for {model_key}")
        
        repo_id = self.find_best_model_variant(model_key)
        model_dir = self.cache_dir / model_key
        
        # Check if already downloaded
        if model_dir.exists() and not force_download:
            logger.info(f"Model {model_key} already exists at {model_dir}")
            return self._get_model_metadata(model_key, repo_id, model_dir)
        
        try:
            # Special handling for SmolVLM to download only the 1.1GB Q4_K_M file
            if model_key == "smolvlm-instruct" and "GGUF" in repo_id:
                logger.info(f"Selective download for SmolVLM - targeting only 1.1GB Q4_K_M file")
                return self._download_smolvlm_selective(repo_id, model_dir)
            
            # Standard download for other models
            logger.info(f"Downloading {repo_id} to {model_dir}")
            
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Successfully downloaded {repo_id}")
            return self._get_model_metadata(model_key, repo_id, Path(downloaded_path))
            
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise
    
    def _download_smolvlm_selective(self, repo_id: str, model_dir: Path) -> Dict:
        """Download only the 1.1GB Q4_K_M GGUF file for SmolVLM"""
        from huggingface_hub import hf_hub_download, list_repo_files
        
        try:
            # Create model directory
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # List all files in the repository
            files = list_repo_files(repo_id)
            logger.info(f"Found {len(files)} files in {repo_id}")
            
            # Find the 1.1GB Q4_K_M GGUF file
            target_file = None
            for file in files:
                if file.endswith('.gguf') and 'q4_k_m' in file.lower():
                    logger.info(f"Found potential target file: {file}")
                    target_file = file
                    break
            
            if not target_file:
                raise ValueError("Could not find Q4_K_M GGUF file in repository")
            
            # Download the specific file
            logger.info(f"📥 Downloading {target_file}")
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=target_file,
                local_dir=str(model_dir),
                resume_download=True
            )
            
            # Also download essential files if they exist
            essential_files = ['README.md', 'config.json', 'tokenizer.json', 'tokenizer_config.json']
            for essential_file in essential_files:
                if essential_file in files:
                    try:
                        logger.info(f"📋 Downloading essential file: {essential_file}")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=essential_file,
                            local_dir=str(model_dir),
                            resume_download=True
                        )
                    except Exception as e:
                        logger.warning(f"Could not download {essential_file}: {e}")
            
            logger.info(f"✅ SmolVLM selective download completed to {model_dir}")
            return self._get_model_metadata("smolvlm-instruct", repo_id, model_dir)
            
        except Exception as e:
            logger.error(f"❌ SmolVLM selective download failed: {e}")
            raise
    
    def _get_model_metadata(self, model_key: str, repo_id: str, model_path: Path) -> Dict:
        """Get metadata about downloaded model"""
        size_bytes = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        size_gb = size_bytes / (1024**3)
        
        return {
            "model_key": model_key,
            "repo_id": repo_id,
            "local_path": str(model_path),
            "size_gb": size_gb,
            "model_type": self.models_config[model_key]["type"],
            "files": [str(f.relative_to(model_path)) for f in model_path.rglob('*') if f.is_file()]
        }
    
    def download_all_models(self, force_download: bool = False) -> Dict:
        """Download all configured models"""
        results = {}
        
        for model_key in self.models_config.keys():
            try:
                results[model_key] = self.download_model(model_key, force_download)
            except Exception as e:
                logger.error(f"Failed to download {model_key}: {e}")
                results[model_key] = {"error": str(e)}
        
        # Save results
        results_file = self.cache_dir / "download_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Download results saved to {results_file}")
        return results
    
    def get_system_info(self) -> Dict:
        """Get system information for the report"""
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        return info

def main():
    """Main function to download all models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("--cache-dir", default="./models", help="Directory to store models")
    parser.add_argument("--force", action="store_true", help="Force re-download existing models")
    parser.add_argument("--token", help="Hugging Face token for private models")
    
    args = parser.parse_args()
    
    # Login if token provided
    if args.token:
        login(token=args.token)
    
    # Initialize downloader
    downloader = ModelDownloader(cache_dir=args.cache_dir)
    
    # Print system info
    system_info = downloader.get_system_info()
    logger.info("System Information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    # Download models
    logger.info("Starting model downloads...")
    results = downloader.download_all_models(force_download=args.force)
    
    # Print summary
    logger.info("\n=== Download Summary ===")
    total_size = 0
    for model_key, result in results.items():
        if "error" in result:
            logger.error(f"{model_key}: ERROR - {result['error']}")
        else:
            size = result['size_gb']
            total_size += size
            logger.info(f"{model_key}: {result['repo_id']} ({size:.2f} GB)")
    
    logger.info(f"Total downloaded size: {total_size:.2f} GB")

if __name__ == "__main__":
    main()