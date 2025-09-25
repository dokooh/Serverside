#!/usr/bin/env python3
"""
Test script to check repository access and GGUF file availability
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
            logger.info(f"✅ Repository {repo_id} is publicly accessible")
            return True
        elif response.status_code == 401:
            logger.warning(f"🔒 Repository {repo_id} requires authentication")
            return False
        else:
            logger.error(f"❌ Repository {repo_id} returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error accessing {repo_id}: {e}")
        return False

def main():
    repositories = [
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    ]
    
    logger.info("🔍 Checking repository accessibility...")
    
    for repo in repositories:
        logger.info(f"Checking {repo}...")
        is_accessible = check_repository_public(repo)
        
        if not is_accessible:
            # Try alternative repositories
            if "Llama-3.2-1B" in repo:
                alternatives = [
                    "microsoft/Llama-3.2-1B-Instruct-GGUF", 
                    "unsloth/Llama-3.2-1B-Instruct-GGUF"
                ]
            elif "TinyLlama" in repo:
                alternatives = [
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Original repo without GGUF
                    "microsoft/TinyLlama-1.1B-Chat-v1.0-GGUF"
                ]
            else:
                alternatives = []
            
            logger.info(f"Trying alternatives for {repo}...")
            for alt in alternatives:
                logger.info(f"  Checking alternative: {alt}")
                if check_repository_public(alt):
                    logger.info(f"  ✅ Alternative found: {alt}")
                    break
            else:
                logger.warning(f"  ❌ No accessible alternatives found for {repo}")

if __name__ == "__main__":
    main()