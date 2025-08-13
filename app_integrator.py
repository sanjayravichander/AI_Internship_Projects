"""
🚀 AI INTERNSHIP PROJECTS - APPLICATION INTEGRATOR
==================================================

Integrates usage management and environment handling with existing applications.
Provides seamless integration without modifying original application code.

Author: AI Intern
Version: 1.0.0 - Public Deployment Edition
"""

import streamlit as st
import sys
import os
from pathlib import Path
import importlib.util
import types
from typing import Any, Dict, Optional
import traceback

# Import our management systems
from usage_manager import get_usage_manager, check_api_usage, record_api_usage
from env_manager import env_manager, get_api_key, is_feature_available

class ApplicationIntegrator:
    """
    Integrates usage management and environment handling with existing applications.
    """
    
    def __init__(self):
        self.original_functions = {}
        self.integration_active = False
    
    def integrate_usage_management(self):
        """Integrate usage management into common API functions."""
        if self.integration_active:
            return
        
        try:
            # Store original functions
            self._patch_groq_calls()
            self._patch_openai_calls()
            self._patch_email_functions()
            self._patch_weather_calls()
            
            self.integration_active = True
            
        except Exception as e:
            st.warning(f"⚠️ Usage management integration partially failed: {e}")
    
    def _patch_groq_calls(self):
        """Patch Groq API calls to include usage management."""
        try:
            # This will be called before any Groq API request
            def groq_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    if not check_api_usage('groq'):
                        # Return a demo response or raise an exception
                        return self._get_demo_response('groq')
                    
                    try:
                        result = original_func(*args, **kwargs)
                        record_api_usage('groq', True)
                        return result
                    except Exception as e:
                        record_api_usage('groq', False)
                        raise e
                
                return wrapper
            
            # Store this wrapper for use by applications
            st.session_state['groq_wrapper'] = groq_wrapper
            
        except Exception as e:
            st.warning(f"⚠️ Groq integration warning: {e}")
    
    def _patch_openai_calls(self):
        """Patch OpenAI API calls to include usage management."""
        try:
            def openai_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    if not check_api_usage('openai'):
                        return self._get_demo_response('openai')
                    
                    try:
                        result = original_func(*args, **kwargs)
                        record_api_usage('openai', True)
                        return result
                    except Exception as e:
                        record_api_usage('openai', False)
                        raise e
                
                return wrapper
            
            st.session_state['openai_wrapper'] = openai_wrapper
            
        except Exception as e:
            st.warning(f"⚠️ OpenAI integration warning: {e}")
    
    def _patch_email_functions(self):
        """Patch email functions to include usage management."""
        try:
            def email_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    if not check_api_usage('email'):
                        st.info("📧 Email sending is temporarily limited. Demo mode active.")
                        return {"status": "demo", "message": "Email would be sent in full version"}
                    
                    try:
                        result = original_func(*args, **kwargs)
                        record_api_usage('email', True)
                        return result
                    except Exception as e:
                        record_api_usage('email', False)
                        raise e
                
                return wrapper
            
            st.session_state['email_wrapper'] = email_wrapper
            
        except Exception as e:
            st.warning(f"⚠️ Email integration warning: {e}")
    
    def _patch_weather_calls(self):
        """Patch weather API calls to include usage management."""
        try:
            def weather_wrapper(original_func):
                def wrapper(*args, **kwargs):
                    if not check_api_usage('weather'):
                        return self._get_demo_response('weather')
                    
                    try:
                        result = original_func(*args, **kwargs)
                        record_api_usage('weather', True)
                        return result
                    except Exception as e:
                        record_api_usage('weather', False)
                        raise e
                
                return wrapper
            
            st.session_state['weather_wrapper'] = weather_wrapper
            
        except Exception as e:
            st.warning(f"⚠️ Weather integration warning: {e}")
    
    def _get_demo_response(self, service: str) -> Dict[str, Any]:
        """Get demo responses when usage limits are reached."""
        demo_responses = {
            'groq': {
                'choices': [{
                    'message': {
                        'content': "🎯 **Demo Mode**: This is a sample AI response. In the full version, you would get real AI-generated content here. The usage limit for this session has been reached to ensure fair access for all users."
                    }
                }]
            },
            'openai': {
                'choices': [{
                    'message': {
                        'content': "🎯 **Demo Mode**: OpenAI response limit reached. This is a sample response to demonstrate the functionality."
                    }
                }]
            },
            'weather': {
                'weather': [{
                    'main': 'Demo',
                    'description': 'Sample weather data - usage limit reached'
                }],
                'main': {
                    'temp': 22.5,
                    'humidity': 65
                },
                'name': 'Demo City'
            }
        }
        
        return demo_responses.get(service, {"status": "demo", "message": "Demo response"})
    
    def setup_environment_for_app(self, app_name: str) -> Dict[str, Any]:
        """Setup environment variables and API keys for a specific app."""
        config = {
            'api_keys': {},
            'features_available': {},
            'demo_mode': False
        }
        
        # Get API keys
        for service in ['groq', 'openai', 'huggingface', 'weather']:
            api_key = get_api_key(service)
            if api_key:
                config['api_keys'][service] = api_key
                config['features_available'][service] = True
            else:
                config['features_available'][service] = False
        
        # Check if we should enable demo mode
        if not config['features_available'].get('groq', False):
            config['demo_mode'] = True
            st.warning("⚠️ Running in demo mode - some features may be limited")
        
        return config
    
    def inject_usage_helpers(self):
        """Inject usage management helper functions into the global namespace."""
        # Make usage functions available to all applications
        import builtins
        
        # Add our helper functions to builtins so they're available everywhere
        builtins.check_api_usage = check_api_usage
        builtins.record_api_usage = record_api_usage
        builtins.get_api_key = get_api_key
        builtins.is_feature_available = is_feature_available
        
        # Add environment manager
        builtins.env_manager = env_manager
        builtins.usage_manager = get_usage_manager()
    
    def create_safe_execution_environment(self, app_path: Path) -> Dict[str, Any]:
        """Create a safe execution environment for applications."""
        # Change to app directory
        original_cwd = os.getcwd()
        if app_path.exists():
            os.chdir(app_path.parent)
        
        # Setup environment
        app_config = self.setup_environment_for_app(app_path.name)
        
        # Inject helpers
        self.inject_usage_helpers()
        
        # Integrate usage management
        self.integrate_usage_management()
        
        return {
            'original_cwd': original_cwd,
            'app_config': app_config,
            'app_path': app_path
        }
    
    def cleanup_execution_environment(self, env_info: Dict[str, Any]):
        """Cleanup after application execution."""
        # Restore original working directory
        os.chdir(env_info['original_cwd'])
    
    def execute_application_safely(self, app_path: str, app_file: str, app_name: str) -> bool:
        """Execute an application with full integration and error handling."""
        try:
            # Setup paths
            base_path = Path("c:/Users/DELL/AI_Internship_Projects")
            full_path = base_path / app_path / app_file
            
            if not full_path.exists():
                st.error(f"❌ Application file not found: {full_path}")
                return False
            
            # Create safe execution environment
            env_info = self.create_safe_execution_environment(full_path)
            
            try:
                # Show loading progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text(f"🚀 Loading {app_name}...")
                progress_bar.progress(25)
                
                # Add app directory to Python path
                app_dir = str(full_path.parent)
                if app_dir not in sys.path:
                    sys.path.insert(0, app_dir)
                
                status_text.text(f"⚙️ Configuring {app_name}...")
                progress_bar.progress(50)
                
                # Read and execute the application
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                status_text.text(f"🎯 Starting {app_name}...")
                progress_bar.progress(75)
                
                # Create a module-like namespace
                module_namespace = {
                    '__name__': '__main__',
                    '__file__': str(full_path),
                    'st': st,
                    'os': os,
                    'sys': sys,
                    'Path': Path,
                    # Add our integrated functions
                    'check_api_usage': check_api_usage,
                    'record_api_usage': record_api_usage,
                    'get_api_key': get_api_key,
                    'is_feature_available': is_feature_available,
                }
                
                # Execute the application code
                exec(content, module_namespace)
                
                progress_bar.progress(100)
                status_text.text(f"✅ {app_name} loaded successfully!")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                return True
                
            except Exception as e:
                st.error(f"❌ Error executing {app_name}: {str(e)}")
                
                # Show detailed error in debug mode
                debug_mode = env_manager.get_secret('DEBUG_MODE', 'false')
                if str(debug_mode).lower() == 'true':
                    st.code(traceback.format_exc())
                
                return False
                
            finally:
                # Always cleanup
                self.cleanup_execution_environment(env_info)
        
        except Exception as e:
            st.error(f"❌ Critical error setting up {app_name}: {str(e)}")
            return False

# Global integrator instance
app_integrator = ApplicationIntegrator()

def execute_app_with_integration(app_path: str, app_file: str, app_name: str) -> bool:
    """
    Execute an application with full integration support.
    
    Args:
        app_path: Path to the application directory
        app_file: Application file name
        app_name: Display name of the application
        
    Returns:
        bool: True if execution was successful
    """
    return app_integrator.execute_application_safely(app_path, app_file, app_name)

def setup_integrated_environment():
    """Setup the integrated environment for all applications."""
    app_integrator.integrate_usage_management()
    app_integrator.inject_usage_helpers()
    
    # Display integration status
    debug_mode = env_manager.get_secret('DEBUG_MODE', 'false')
    if str(debug_mode).lower() == 'true':
        st.sidebar.success("🔧 Integration: Active")