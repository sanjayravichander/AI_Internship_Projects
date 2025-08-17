"""
🚀 AI INTERNSHIP PROJECTS - HUGGING FACE SPACES DEPLOYMENT
=========================================================

Main entry point for Hugging Face Spaces deployment.
This file imports and runs the master enterprise dashboard.

Author: AI Intern
Version: 3.0.0 - Hugging Face Spaces Edition
"""

import os
import sys
import streamlit as st

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the master application
try:
    from master_app_enterprise import main
    
    if __name__ == "__main__":
        # Set environment variables for Hugging Face Spaces
        os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
        os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
        os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
        
        # Run the main application
        main()
        
except ImportError as e:
    st.error(f"Failed to import master application: {e}")
    st.info("Please ensure all dependencies are installed correctly.")
except Exception as e:
    st.error(f"Application startup error: {e}")
    st.info("Please check the logs for more details.")