#!/usr/bin/env python3
"""
Setup script for Hugging Face Spaces deployment
Handles model downloads and environment setup
"""

import subprocess
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command with error handling"""
    try:
        logger.info(f"Running: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def download_spacy_models():
    """Download required spaCy models"""
    models = [
        "en_core_web_sm",
        "en_core_web_md"  # Backup model
    ]
    
    for model in models:
        success = run_command(
            f"python -m spacy download {model}",
            f"Downloading spaCy model: {model}"
        )
        if success:
            logger.info(f"Successfully downloaded {model}")
            break
    else:
        logger.warning("Failed to download any spaCy models")

def setup_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk_downloads = [
            'punkt',
            'stopwords',
            'vader_lexicon',
            'wordnet',
            'averaged_perceptron_tagger'
        ]
        
        for item in nltk_downloads:
            try:
                nltk.download(item, quiet=True)
                logger.info(f"Downloaded NLTK data: {item}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK data {item}: {e}")
                
    except ImportError:
        logger.warning("NLTK not available, skipping NLTK data download")

def create_directories():
    """Create necessary directories"""
    directories = [
        "temp",
        "uploads",
        "downloads",
        ".streamlit"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to create directory {directory}: {e}")

def setup_streamlit_config():
    """Create Streamlit configuration"""
    config_content = """
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#007bff"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#2c3e50"
"""
    
    try:
        with open(".streamlit/config.toml", "w") as f:
            f.write(config_content.strip())
        logger.info("Created Streamlit configuration")
    except Exception as e:
        logger.error(f"Failed to create Streamlit config: {e}")

def main():
    """Main setup function"""
    logger.info("Starting Hugging Face Spaces setup...")
    
    # Create necessary directories
    create_directories()
    
    # Setup Streamlit configuration
    setup_streamlit_config()
    
    # Download spaCy models
    download_spacy_models()
    
    # Setup NLTK data
    setup_nltk_data()
    
    logger.info("Setup completed successfully!")

if __name__ == "__main__":
    main()