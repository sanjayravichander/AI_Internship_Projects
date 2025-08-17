"""
Rate Limit Handler for Groq API
Handles rate limiting and API errors gracefully
"""

import time
import streamlit as st
from functools import wraps
import logging

def handle_groq_error(func):
    """Decorator to handle Groq API errors and rate limiting"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle rate limiting
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        st.warning(f"⏳ Rate limit hit. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        st.error("❌ Rate limit exceeded. Please try again later.")
                        return None
                
                # Handle other API errors
                elif "api" in error_msg or "key" in error_msg:
                    st.error(f"❌ API Error: {e}")
                    return None
                
                # Handle network errors
                elif "connection" in error_msg or "timeout" in error_msg:
                    if attempt < max_retries - 1:
                        st.warning(f"🔄 Connection issue. Retrying... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        st.error("❌ Connection failed. Please check your internet connection.")
                        return None
                
                # Handle other errors
                else:
                    st.error(f"❌ Unexpected error: {e}")
                    return None
        
        return None
    
    return wrapper

def check_api_key():
    """Check if Groq API key is available"""
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        st.info("💡 Please add your Groq API key to the .env file")
        return False
    return True