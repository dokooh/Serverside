#!/usr/bin/env python3
"""
Test script to verify the new model repositories are accessible
"""

import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_repository_public(repo_id):
    """Check if a repository is publicly accessible"""
    try:
        url = f"https://huggingface.co/api/models/{repo_id}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            model_size = data.get('safetensors', {}).get('total', 0)
            logger.info(f"✅ {repo_id} is publicly accessible")
            if model_size > 0:
                size_gb = model_size / (1024**3)
                logger.info(f"   Model size: {size_gb:.2f} GB")
            return True
        elif response.status_code == 401:
            logger.warning(f"🔒 {repo_id} requires authentication")
            return False
        else:
            logger.error(f"❌ {repo_id} returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error accessing {repo_id}: {e}")
        return False

def main():
    repositories = [
        "unsloth/Llama-3.2-1B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    
    logger.info("🔍 Checking new model repositories...")
    
    all_accessible = True
    for repo in repositories:
        logger.info(f"Checking {repo}...")
        is_accessible = check_repository_public(repo)
        if not is_accessible:
            all_accessible = False
    
    if all_accessible:
        logger.info("✅ All repositories are publicly accessible!")
        logger.info("✅ Models can be downloaded without authentication")
    else:
        logger.error("❌ Some repositories require authentication")

if __name__ == "__main__":
    main()