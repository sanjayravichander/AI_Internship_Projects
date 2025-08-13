"""
🚀 AI INTERNSHIP PROJECTS - ENVIRONMENT MANAGER
===============================================

Secure environment variable and API key management for public deployment.
Handles both local development and cloud deployment configurations.

Author: AI Intern
Version: 1.0.0 - Public Deployment Edition
"""

import streamlit as st
import os
from typing import Optional, Dict, Any
import warnings

class EnvironmentManager:
    """
    Manages environment variables and API keys securely for different deployment modes.
    """
    
    def __init__(self):
        self.deployment_mode = self._detect_deployment_mode()
        self._load_environment()
    
    def _safe_bool_convert(self, value, default=False):
        """Safely convert a value to boolean, handling both string and bool types."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if value is None:
            return default
        # Handle any other type by converting to string first
        try:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        except:
            return default
    
    def _detect_deployment_mode(self) -> str:
        """Detect if we're running locally or in cloud deployment."""
        if os.getenv('STREAMLIT_SHARING') or os.getenv('STREAMLIT_CLOUD'):
            return 'cloud'
        elif hasattr(st, 'secrets') and st.secrets:
            return 'cloud'
        else:
            return 'local'
    
    def _load_environment(self):
        """Load environment variables based on deployment mode."""
        if self.deployment_mode == 'cloud':
            self._load_cloud_secrets()
        else:
            self._load_local_env()
    
    def _load_cloud_secrets(self):
        """Load secrets from Streamlit Cloud secrets management."""
        try:
            # Streamlit Cloud secrets are available via st.secrets
            self.secrets = dict(st.secrets) if hasattr(st, 'secrets') and st.secrets else {}
            
            # Validate required secrets
            required_keys = ['GROQ_API_KEY']
            missing_keys = [key for key in required_keys if not self.get_secret(key)]
            
            if missing_keys:
                st.error(f"""
                🚨 **Missing Required Secrets**
                
                The following secrets are not configured in Streamlit Cloud:
                {', '.join(missing_keys)}
                
                Please add these secrets in your Streamlit Cloud app settings.
                """)
                st.stop()
                
        except Exception as e:
            st.error(f"""
            🚨 **Cloud Secrets Error**
            
            Failed to load secrets from Streamlit Cloud: {e}
            
            Please check your app's secret configuration.
            """)
            st.stop()
    
    def _load_local_env(self):
        """Load environment variables from .env file for local development."""
        try:
            # Try to load from .env file
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                from dotenv import load_dotenv
                load_dotenv(env_path)
            
            # Check for required environment variables
            required_keys = ['GROQ_API_KEY']
            missing_keys = [key for key in required_keys if not os.getenv(key)]
            
            if missing_keys:
                st.warning(f"""
                ⚠️ **Missing Environment Variables**
                
                The following environment variables are not set:
                {', '.join(missing_keys)}
                
                Some features may not work properly. Please check your .env file.
                """)
                
        except ImportError:
            st.warning("""
            ⚠️ **python-dotenv not installed**
            
            For local development, install python-dotenv:
            ```bash
            pip install python-dotenv
            ```
            """)
        except Exception as e:
            st.warning(f"⚠️ Environment loading error: {e}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret/environment variable safely.
        
        Args:
            key: The secret key to retrieve
            default: Default value if key is not found
            
        Returns:
            The secret value or default
        """
        if self.deployment_mode == 'cloud':
            return self.secrets.get(key, default)
        else:
            return os.getenv(key, default)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            service: Service name ('groq', 'openai', 'huggingface', etc.)
            
        Returns:
            API key or None if not available
        """
        key_mapping = {
            'groq': 'GROQ_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'huggingface': 'HUGGING_FACE_API_KEY',
            'weather': 'WEATHER_API_KEY'
        }
        
        env_key = key_mapping.get(service.lower())
        if not env_key:
            return None
        
        api_key = self.get_secret(env_key)
        
        # Validate API key format (basic check)
        if api_key and self._validate_api_key(service, api_key):
            return api_key
        
        return None
    
    def _validate_api_key(self, service: str, api_key: str) -> bool:
        """
        Basic validation of API key format.
        
        Args:
            service: Service name
            api_key: API key to validate
            
        Returns:
            True if key format looks valid
        """
        if not api_key or api_key.startswith('your_') or api_key == 'your_api_key_here':
            return False
        
        # Service-specific validation
        validation_rules = {
            'groq': lambda k: k.startswith('gsk_') and len(k) > 20,
            'openai': lambda k: k.startswith('sk-') and len(k) > 20,
            'huggingface': lambda k: k.startswith('hf_') and len(k) > 20,
            'weather': lambda k: len(k) > 10  # OpenWeatherMap keys are typically 32 chars
        }
        
        validator = validation_rules.get(service.lower())
        if validator:
            return validator(api_key)
        
        return len(api_key) > 10  # Generic validation
    
    def get_email_config(self) -> Dict[str, Optional[str]]:
        """Get email configuration for voice assistant."""
        return {
            'email': self.get_secret('EMAIL_ADDRESS'),
            'password': self.get_secret('EMAIL_PASSWORD')
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration."""
        return {
            'groq_model': self.get_secret('GROQ_MODEL', 'llama-3.3-70b-versatile'),
            'groq_temperature': float(self.get_secret('GROQ_TEMPERATURE', '0.7')),
            'groq_max_tokens': int(self.get_secret('GROQ_MAX_TOKENS', '1000')),
            'embedding_model': self.get_secret('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled based on available API keys.
        
        Args:
            feature: Feature name ('email', 'weather', 'openai', etc.)
            
        Returns:
            True if feature is enabled and has required API keys
        """
        feature_requirements = {
            'email': ['EMAIL_ADDRESS', 'EMAIL_PASSWORD'],
            'weather': ['WEATHER_API_KEY'],
            'openai': ['OPENAI_API_KEY'],
            'groq': ['GROQ_API_KEY'],
            'huggingface': ['HUGGING_FACE_API_KEY']
        }
        
        required_keys = feature_requirements.get(feature.lower(), [])
        return all(self.get_secret(key) for key in required_keys)
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information for debugging."""
        return {
            'deployment_mode': self.deployment_mode,
            'features_enabled': {
                'groq': self.is_feature_enabled('groq'),
                'openai': self.is_feature_enabled('openai'),
                'email': self.is_feature_enabled('email'),
                'weather': self.is_feature_enabled('weather'),
                'huggingface': self.is_feature_enabled('huggingface')
            },
            'debug_mode': self._safe_bool_convert(self.get_secret('DEBUG_MODE', 'false'))
        }
    
    def display_deployment_status(self):
        """Display deployment status in the sidebar for debugging."""
        if self._safe_bool_convert(self.get_secret('DEBUG_MODE', 'false')):
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔧 Deployment Status")
            
            info = self.get_deployment_info()
            st.sidebar.caption(f"Mode: {info['deployment_mode'].title()}")
            
            for feature, enabled in info['features_enabled'].items():
                status = "✅" if enabled else "❌"
                st.sidebar.caption(f"{status} {feature.title()}")

# Global environment manager instance
env_manager = EnvironmentManager()

def get_api_key(service: str) -> Optional[str]:
    """
    Convenient function to get API keys.
    
    Args:
        service: Service name ('groq', 'openai', 'huggingface', 'weather')
        
    Returns:
        API key or None
    """
    return env_manager.get_api_key(service)

def is_feature_available(feature: str) -> bool:
    """
    Check if a feature is available (has required API keys).
    
    Args:
        feature: Feature name
        
    Returns:
        True if feature is available
    """
    return env_manager.is_feature_enabled(feature)

def get_model_config() -> Dict[str, Any]:
    """Get AI model configuration."""
    return env_manager.get_model_config()

def show_api_key_info():
    """Show API key information to users."""
    if not env_manager.is_feature_enabled('groq'):
        st.error("""
        🚨 **API Configuration Required**
        
        This application requires API keys to function properly.
        
        **For Streamlit Cloud Deployment:**
        - Configure secrets in your Streamlit Cloud app settings
        - Add at minimum: `GROQ_API_KEY`
        
        **For Local Development:**
        - Create a `.env` file with your API keys
        - See `.env.example` for the required format
        """)
        return False
    
    return True