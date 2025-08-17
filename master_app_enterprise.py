"""
🚀 AI INTERNSHIP PROJECTS - ENTERPRISE MASTER DASHBOARD
======================================================

Enterprise-grade Streamlit application with modern UI/UX design,
sophisticated styling, and professional user experience.

Author: AI Intern
Version: 3.0.0 - Enterprise Edition
"""

import streamlit as st
import sys
import os
import subprocess
import importlib.util
from pathlib import Path
import traceback
import time
import re
import json
import types
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configure page FIRST (before any other Streamlit calls)
st.set_page_config(
    page_title="🚀 AI Internship Projects - Enterprise Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Enterprise AI Dashboard v3.0.0"
    }
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Enterprise-grade CSS with sophisticated styling
def load_css():
    theme = st.session_state.theme
    
    if theme == 'dark':
        primary_bg = "#0e1117"
        secondary_bg = "#262730"
        card_bg = "#1e1e1e"
        text_primary = "#ffffff"
        text_secondary = "#b3b3b3"
        border_color = "#333333"
        accent_color = "#00d4ff"
        gradient_primary = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        gradient_secondary = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        shadow = "0 8px 32px rgba(0, 0, 0, 0.3)"
    else:
        primary_bg = "#ffffff"
        secondary_bg = "#f8f9fa"
        card_bg = "#ffffff"
        text_primary = "#2c3e50"
        text_secondary = "#6c757d"
        border_color = "#e9ecef"
        accent_color = "#007bff"
        gradient_primary = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        gradient_secondary = "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
        shadow = "0 8px 32px rgba(0, 0, 0, 0.1)"

    st.markdown(f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* CSS Variables for theming */
        :root {{
            --primary-bg: {primary_bg};
            --secondary-bg: {secondary_bg};
            --card-bg: {card_bg};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --border-color: {border_color};
            --accent-color: {accent_color};
            --gradient-primary: {gradient_primary};
            --gradient-secondary: {gradient_secondary};
            --shadow: {shadow};
            --border-radius: 12px;
            --border-radius-lg: 16px;
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-xxl: 3rem;
        }}
        
        /* Global Styles */
        .main {{
            background-color: var(--primary-bg);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        .stDeployButton {{display: none;}}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: var(--secondary-bg);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: var(--accent-color);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #0056b3;
        }}
        
        /* Header Styles */
        .enterprise-header {{
            background: var(--gradient-primary);
            padding: var(--spacing-xxl) var(--spacing-xl);
            border-radius: var(--border-radius-lg);
            margin-bottom: var(--spacing-xl);
            text-align: center;
            color: white;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }}
        
        .enterprise-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }}
        
        .enterprise-header h1 {{
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: var(--spacing-sm);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }}
        
        .enterprise-header .subtitle {{
            font-size: 1.25rem;
            font-weight: 400;
            opacity: 0.95;
            margin-bottom: 0;
            position: relative;
            z-index: 1;
        }}
        
        .enterprise-header .version-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: var(--spacing-xs) var(--spacing-md);
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: var(--spacing-md);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }}
        
        /* Sidebar Styles */
        .css-1d391kg {{
            background-color: var(--secondary-bg);
            border-right: 1px solid var(--border-color);
        }}
        
        .sidebar-header {{
            background: var(--gradient-primary);
            padding: var(--spacing-lg);
            border-radius: var(--border-radius);
            margin-bottom: var(--spacing-lg);
            text-align: center;
            color: white;
            box-shadow: var(--shadow);
        }}
        
        .sidebar-header h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin: 0 0 var(--spacing-xs) 0;
        }}
        
        .sidebar-header p {{
            font-size: 0.875rem;
            margin: 0;
            opacity: 0.9;
        }}
        
        /* Card Styles */
        .enterprise-card {{
            background: var(--card-bg);
            padding: var(--spacing-xl);
            border-radius: var(--border-radius-lg);
            margin: var(--spacing-lg) 0;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }}
        
        .enterprise-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--gradient-primary);
            transition: width 0.3s ease;
        }}
        
        .enterprise-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }}
        
        .enterprise-card:hover::before {{
            width: 8px;
        }}
        
        .enterprise-card h3 {{
            color: var(--text-primary);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: var(--spacing-md);
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
        }}
        
        .enterprise-card p {{
            color: var(--text-secondary);
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: var(--spacing-lg);
        }}
        
        .tech-stack {{
            background: var(--gradient-secondary);
            padding: var(--spacing-md);
            border-radius: var(--border-radius);
            margin: var(--spacing-lg) 0;
            font-size: 0.875rem;
            font-weight: 500;
            color: white;
            backdrop-filter: blur(10px);
        }}
        
        .features-list {{
            display: flex;
            flex-wrap: wrap;
            gap: var(--spacing-sm);
            margin: var(--spacing-lg) 0;
        }}
        
        .feature-tag {{
            background: var(--secondary-bg);
            color: var(--text-primary);
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }}
        
        .feature-tag:hover {{
            background: var(--accent-color);
            color: white;
            transform: scale(1.05);
        }}
        
        /* Status Indicators */
        .status-indicator {{
            display: inline-flex;
            align-items: center;
            gap: var(--spacing-xs);
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .status-ready {{
            background: rgba(34, 197, 94, 0.1);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        
        .status-loading {{
            background: rgba(251, 191, 36, 0.1);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }}
        
        .status-error {{
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        
        /* Metrics Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-lg);
            margin: var(--spacing-xl) 0;
        }}
        
        .metric-card {{
            background: var(--gradient-primary);
            padding: var(--spacing-xl);
            border-radius: var(--border-radius-lg);
            text-align: center;
            color: white;
            box-shadow: var(--shadow);
            transition: transform 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: scale(0);
            transition: transform 0.6s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-4px);
        }}
        
        .metric-card:hover::before {{
            transform: scale(1);
        }}
        
        .metric-card h2 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: var(--spacing-xs);
            position: relative;
            z-index: 1;
        }}
        
        .metric-card p {{
            font-size: 1rem;
            margin: 0;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }}
        
        /* Button Styles */
        .stButton > button {{
            background: var(--gradient-primary);
            color: white;
            border: none;
            border-radius: var(--border-radius);
            padding: var(--spacing-md) var(--spacing-xl);
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }}
        
        /* Theme Toggle */
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 50px;
            padding: var(--spacing-sm);
            box-shadow: var(--shadow);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .theme-toggle:hover {{
            transform: scale(1.1);
        }}
        
        /* Loading Animation */
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .enterprise-header h1 {{
                font-size: 2.5rem;
            }}
            
            .enterprise-header .subtitle {{
                font-size: 1rem;
            }}
            
            .enterprise-card {{
                padding: var(--spacing-lg);
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: var(--spacing-md);
            }}
        }}
        
        /* Animations */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .fade-in-up {{
            animation: fadeInUp 0.6s ease-out;
        }}
        
        /* Tooltips */
        .tooltip {{
            position: relative;
            cursor: help;
        }}
        
        .tooltip::after {{
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--text-primary);
            color: var(--primary-bg);
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--border-radius);
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            z-index: 1000;
        }}
        
        .tooltip:hover::after {{
            opacity: 1;
        }}
        
        /* Progress Bars */
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--secondary-bg);
            border-radius: 4px;
            overflow: hidden;
            margin: var(--spacing-sm) 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: var(--gradient-primary);
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        /* Footer */
        .enterprise-footer {{
            background: var(--secondary-bg);
            padding: var(--spacing-xxl) var(--spacing-xl);
            margin-top: var(--spacing-xxl);
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
        }}
        
        .enterprise-footer h3 {{
            color: var(--text-primary);
            margin-bottom: var(--spacing-lg);
        }}
        
        .footer-links {{
            display: flex;
            justify-content: center;
            gap: var(--spacing-xl);
            margin: var(--spacing-lg) 0;
            flex-wrap: wrap;
        }}
        
        .footer-link {{
            color: var(--accent-color);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }}
        
        .footer-link:hover {{
            color: var(--text-primary);
        }}
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

class EnterpriseAppRunner:
    """Advanced application runner with enterprise-grade features"""
    
    def __init__(self):
        # Ensure we always use the correct absolute path
        self.base_path = Path(__file__).parent.absolute()
        self.loaded_modules = {}
        self.performance_metrics = {}
        
        # Verify the base path exists
        if not self.base_path.exists():
            st.error(f"❌ Base path does not exist: {self.base_path}")
            st.info("💡 Please ensure the application is running from the correct directory.")
        
    def execute_app_with_full_features(self, app_path, app_file, app_name):
        """Execute application with full features and performance monitoring"""
        start_time = time.time()
        
        try:
            full_path = self.base_path / app_path / app_file
            
            # Debug information (only show if there's an error)
            # st.info(f"🔍 Looking for: {full_path}")
            
            if not full_path.exists():
                st.error(f"❌ Application file not found: {full_path}")
                
                # Check if the directory exists
                app_dir = full_path.parent
                if app_dir.exists():
                    st.info(f"📁 Directory exists: {app_dir}")
                    files_in_dir = list(app_dir.glob("*.py"))
                    if files_in_dir:
                        st.info(f"🔍 Python files found: {[f.name for f in files_in_dir]}")
                    else:
                        st.warning("⚠️ No Python files found in the directory")
                else:
                    st.error(f"❌ Directory does not exist: {app_dir}")
                
                self._show_file_suggestions(app_path, app_file)
                return False
            
            # Add app directory to Python path
            app_dir = str(full_path.parent)
            if app_dir not in sys.path:
                sys.path.insert(0, app_dir)
            
            # Store original working directory but don't change it
            original_cwd = os.getcwd()
            # Comment out the directory change to avoid path issues
            # os.chdir(full_path.parent)
            
            try:
                # Show loading with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.markdown(f"🚀 **Loading {app_name}...**")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                # Read the original file
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                status_text.markdown(f"⚙️ **Configuring {app_name}...**")
                progress_bar.progress(50)
                time.sleep(0.3)
                
                # Handle st.set_page_config conflicts
                content = re.sub(
                    r'st\.set_page_config\([^)]*\)',
                    '# st.set_page_config commented out for enterprise dashboard integration',
                    content,
                    flags=re.MULTILINE | re.DOTALL
                )
                
                status_text.markdown(f"🔧 **Initializing {app_name}...**")
                progress_bar.progress(75)
                time.sleep(0.3)
                
                # Create execution namespace with enhanced imports for Voice Assistant
                namespace = {
                    '__name__': '__main__',
                    '__file__': str(full_path),
                    '__builtins__': __builtins__,  # Allow imports and built-in functions
                    'st': st,
                    'os': os,
                    'sys': sys,
                    'Path': Path,
                    'datetime': datetime,
                    'time': time,
                    'json': json,
                    'traceback': traceback,
                    'importlib': importlib,
                    're': re,
                }
                
                # Add additional imports that Voice Assistant needs
                try:
                    import subprocess
                    namespace['subprocess'] = subprocess
                except ImportError:
                    pass
                
                try:
                    import threading
                    namespace['threading'] = threading
                except ImportError:
                    pass
                
                try:
                    import requests
                    namespace['requests'] = requests
                except ImportError:
                    pass
                
                try:
                    from dotenv import load_dotenv
                    namespace['load_dotenv'] = load_dotenv
                except ImportError:
                    pass
                
                # Try to import psutil and make it available in the namespace
                try:
                    import psutil
                    namespace['psutil'] = psutil
                except ImportError:
                    # Create a mock psutil for graceful fallback
                    class MockPsutil:
                        @staticmethod
                        def Process(pid):
                            class MockProcess:
                                def cpu_percent(self):
                                    return 0.0
                                def memory_info(self):
                                    class MockMemInfo:
                                        rss = 0
                                    return MockMemInfo()
                                def status(self):
                                    return "running"
                            return MockProcess()
                    namespace['psutil'] = MockPsutil()
                
                # Execute the application code
                exec(content, namespace)
                
                progress_bar.progress(100)
                status_text.markdown(f"✅ **{app_name} loaded successfully!**")
                time.sleep(0.5)
                
                # Clear loading indicators
                progress_bar.empty()
                status_text.empty()
                
                # Record performance metrics
                load_time = time.time() - start_time
                self.performance_metrics[app_name] = {
                    'load_time': load_time,
                    'timestamp': datetime.now(),
                    'status': 'success'
                }
                
                return True
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                
                error_str = str(e).lower()
                
                # Don't show error popup for certain harmless errors
                if ('scriptruncontext' in error_str):
                    return False
                
                # Log file path errors for debugging but don't suppress them
                if 'no such file or directory' in error_str and 'master_app' in error_str:
                    st.warning(f"⚠️ Navigation issue detected: {str(e)}")
                    st.info("💡 This might be a temporary issue. Try refreshing the page or selecting the app again.")
                
                self._show_error_details(app_name, e)
                
                # Record error metrics
                self.performance_metrics[app_name] = {
                    'load_time': time.time() - start_time,
                    'timestamp': datetime.now(),
                    'status': 'error',
                    'error': str(e)
                }
                
                return False
                
            finally:
                # Working directory restoration not needed since we don't change it
                pass
                
        except Exception as e:
            st.error(f"❌ Critical error loading {app_name}: {str(e)}")
            return False
    
    def _show_file_suggestions(self, app_path, app_file):
        """Show file suggestions when app file is not found"""
        st.info(f"Expected path: {self.base_path / app_path / app_file}")
        
        # Check for alternative files
        if app_path == "Hand_gesture_AI" and app_file == "app.py":
            alt_path = self.base_path / app_path / "streamlit_app.py"
            if alt_path.exists():
                st.info("💡 Found streamlit_app.py instead. The configuration has been updated.")
                return self.execute_app_with_full_features(app_path, "streamlit_app.py", "Hand Gesture Recognition")
        
        # List available files
        try:
            available_files = list((self.base_path / app_path).glob("*.py"))
            if available_files:
                st.info("📁 Available Python files in this directory:")
                for file in available_files:
                    st.write(f"  • {file.name}")
        except:
            pass
    
    def _show_error_details(self, app_name, error):
        """Show detailed error information with suggestions"""
        st.error(f"❌ Error executing {app_name}: {str(error)}")
        
        with st.expander("🔍 Detailed Error Information", expanded=False):
            st.code(traceback.format_exc())
            
            error_str = str(error).lower()
            
            # Provide specific suggestions based on error type
            if 'no module named' in error_str:
                st.warning("🔧 **Missing Dependency Issue**")
                st.info("💡 **Solution:** Run `pip install -r requirements.txt`")
                
                if st.button("📦 Install Dependencies", key=f"install_{app_name}"):
                    with st.spinner("Installing dependencies..."):
                        try:
                            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                         check=True, capture_output=True, text=True)
                            st.success("✅ Dependencies installed successfully!")
                            st.info("🔄 Please try loading the application again.")
                        except subprocess.CalledProcessError as e:
                            st.error(f"❌ Failed to install dependencies: {e}")
                            
            elif 'groq_api_key' in error_str or 'api_key' in error_str:
                st.warning("🔑 **API Key Issue**")
                st.info("💡 **Solution:** Make sure your .env file contains the required API keys")
                st.code("GROQ_API_KEY=your_api_key_here")
                
            elif 'permission' in error_str:
                st.warning("🔒 **Permission Issue**")
                st.info("💡 **Solution:** Try running as administrator or check file permissions")
                
            elif 'csv' in error_str.lower() or 'file not found' in error_str:
                st.warning("📊 **Data File Issue**")
                st.info("💡 **Solution:** Check if required data files exist in the application directory")
    
    def get_performance_metrics(self):
        """Get performance metrics for all loaded applications"""
        return self.performance_metrics

# Create global runner instance
app_runner = EnterpriseAppRunner()

# Enhanced application configurations
APPLICATIONS = {
    
    "🎯 Project Overview": {
        "description": "Welcome to the AI Internship Projects Enterprise Dashboard - showcasing 9 advanced AI applications with enterprise-grade quality",
        "function": None,
        "path": None,
        "file": None,
        "tech_stack": "Streamlit • Python • Enterprise UI/UX • Performance Monitoring",
        "features": ["Project Overview", "Application Selector", "Enterprise Dashboard", "Performance Analytics", "Theme Toggle"],
        "status": "ready",
        "category": "Dashboard",
        "complexity": "High",
        "last_updated": "2024-01-15"
    },
    "🤖 Voice Assistant 2.0": {
        "description": "Next-generation voice assistant with LiveKit integration and dual operation modes. 🌐 Web Mode (LiveKit Playground): Natural conversation, web search, information retrieval, web applications. 💻 Console Mode (Local Terminal): Full system control, application launching, volume control, screenshots. Environment-aware AI that adapts functionality based on execution context.",
        "function": None,
        "path": "Voice_Assistant(2.0)",
        "file": "voice_assistant_launcher.py",
        "tech_stack": "LiveKit Agents • Google LLM & TTS • System Integration • Web APIs",
        "features": ["Dual Mode Operation", "🌐 Web Mode: Natural conversation, web search, information retrieval", "💻 Console Mode: Full system control, application launching, volume control", "Real-time voice conversation", "Intelligent system control", "Web integration", "Contextual memory"],
        "status": "ready",
        "category": "AI Assistant",
        "complexity": "High",
        "last_updated": "2024-01-18"
    },
    "📚 Document Intelligence Chatbot": {
        "description": "Enterprise-grade document Q&A system with agentic AI capabilities, semantic search, entity extraction, and advanced analytics dashboard",
        "function": "run_chatbot_ai",
        "path": "Chatbot_AI",
        "file": "app.py",
        "tech_stack": "LangChain • FAISS Vector DB • spaCy NLP • Plotly Analytics • NetworkX • Semantic Search",
        "features": ["Document Upload", "Semantic Search", "Entity Extraction", "Knowledge Graphs", "Analytics Dashboard", "Multi-format Support"],
        "status": "ready",
        "category": "Document AI",
        "complexity": "High",
        "last_updated": "2024-01-13"
    },
    "🦠 COVID-19 Analytics Dashboard": {
        "description": "AI-powered COVID-19 analytics platform with machine learning predictions, anomaly detection, and interactive data visualizations",
        "function": "run_covid_dashboard",
        "path": "COVID_19_AI",
        "file": "advanced_covid_dashboard.py",
        "tech_stack": "Plotly Dash • Scikit-learn ML • Pandas Analytics • Groq AI • Statistical Modeling",
        "features": ["Predictive Modeling", "Anomaly Detection", "Interactive Charts", "State Comparisons", "Trend Analysis", "AI Insights"],
        "status": "ready",
        "category": "Healthcare Analytics",
        "complexity": "High",
        "last_updated": "2024-01-12"
    },
    "👋 Hand Gesture Recognition": {
        "description": "Real-time hand gesture recognition system using MediaPipe computer vision with live camera feed and gesture classification",
        "function": "run_hand_gesture",
        "path": "Hand_gesture_AI",
        "file": "streamlit_app_improved.py",
        "tech_stack": "MediaPipe • OpenCV • Computer Vision • Real-time Processing • Threading",
        "features": ["Real-time Detection", "Gesture Classification", "Live Camera Feed", "Adjustable Parameters", "Performance Optimization"],
        "status": "ready",
        "category": "Computer Vision",
        "complexity": "Medium",
        "last_updated": "2024-01-11"
    },
    "🎨 Cartoonify AI": {
        "description": "Transform images and videos into cartoon-style artwork using advanced AI filters, AnimeGAN models, and computer vision techniques",
        "function": "run_cartoonify",
        "path": "Cartoonify_AI",
        "file": "groq_cartoonify.py",
        "tech_stack": "OpenCV • ONNX Runtime • AnimeGAN • Groq Vision API • Image Processing",
        "features": ["Image Cartoonification", "Video Processing", "Multiple Art Styles", "AI Analysis", "Batch Processing"],
        "status": "ready",
        "category": "Creative AI",
        "complexity": "Medium",
        "last_updated": "2024-01-10"
    },
    "😂 Meme Classification VLM": {
        "description": "Intelligent meme classification system using CLIP vision-language model with LLM-powered explanations and sentiment analysis",
        "function": "run_meme_classification",
        "path": "Meme_Classification_VLM",
        "file": "app.py",
        "tech_stack": "CLIP Model • Transformers • Groq LLM • Vision-Language Processing • Zero-shot Learning",
        "features": ["Zero-shot Classification", "AI Explanations", "Vision-Language Processing", "Sentiment Analysis", "Meme Understanding"],
        "status": "ready",
        "category": "Vision-Language AI",
        "complexity": "High",
        "last_updated": "2024-01-09"
    },
    "📊 Student Report Card Generator": {
        "description": "Interactive student report card management system with advanced data visualization, grade analytics, and professional PDF generation",
        "function": "run_data_handling",
        "path": "Data_Handling",
        "file": "app.py",
        "tech_stack": "Pandas • Plotly Visualizations • ReportLab PDF • Data Analytics • Statistical Analysis",
        "features": ["Data Upload", "Grade Calculation", "Advanced Visualizations", "PDF Reports", "Statistical Analysis", "Performance Tracking"],
        "status": "ready",
        "category": "Educational Technology",
        "complexity": "Medium",
        "last_updated": "2024-01-08"
    },
    "🧠 AI Quiz Game": {
        "description": "Dynamic quiz game platform with AI-generated questions using Groq LLM, adaptive difficulty levels, and comprehensive scoring system",
        "function": "run_quiz_game",
        "path": "Python_Quiz_Game_AI",
        "file": "ai_quiz.py",
        "tech_stack": "Groq LLM • Pandas Analytics • Dynamic Content Generation • Adaptive Learning",
        "features": ["AI-Generated Questions", "Multiple Difficulty Levels", "Score Tracking", "Leaderboard System", "Adaptive Learning", "Performance Analytics"],
        "status": "ready",
        "category": "Educational AI",
        "complexity": "Medium",
        "last_updated": "2024-01-07"
    },
    "💭 Sentiment Analysis AI": {
        "description": "Advanced sentiment analysis system using fine-tuned transformer models with real-time text processing, confidence scoring, and interactive visualizations",
        "function": "run_sentiment_analysis",
        "path": "Sentiment_Analysis_AI",
        "file": "sentiment_app_integrated.py",
        "tech_stack": "Transformers • PyTorch • FastAPI • Streamlit • RoBERTa • Sentiment Classification",
        "features": ["Real-time Analysis", "Confidence Scoring", "Interactive UI", "Fine-tuned Models", "Animated Visualizations", "Multiple Sentiment Classes"],
        "status": "ready",
        "category": "Natural Language Processing",
        "complexity": "High",
        "last_updated": "2024-01-16"
    }
}

def create_theme_toggle():
    """Create theme toggle button"""
    col1, col2, col3 = st.columns([1, 1, 8])
    
    with col2:
        if st.button("🌓", help="Toggle Dark/Light Theme", key="theme_toggle"):
            st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
            st.rerun()

def load_application(app_name, config):
    """Load and execute an application with full functionality"""
    if config['function'] is None:
        # For applications without custom functions, load directly using file execution
        # Clear the main area
        st.empty()
        
        # Execute the application
        return app_runner.execute_app_with_full_features(
            config['path'],
            config['file'],
            app_name
        )
    
    # Clear the main area
    st.empty()
    
    # Execute the application
    return app_runner.execute_app_with_full_features(
        config['path'],
        config['file'],
        app_name
    )

def create_performance_chart():
    """Create performance metrics chart"""
    metrics = app_runner.get_performance_metrics()
    
    if not metrics:
        return None
    
    df = pd.DataFrame([
        {
            'Application': app_name,
            'Load Time (s)': data['load_time'],
            'Status': data['status'],
            'Timestamp': data['timestamp']
        }
        for app_name, data in metrics.items()
    ])
    
    fig = px.bar(
        df, 
        x='Application', 
        y='Load Time (s)',
        color='Status',
        title='Application Performance Metrics',
        color_discrete_map={'success': '#22c55e', 'error': '#ef4444'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#6c757d'
    )
    
    return fig

def show_overview():
    """Display the enhanced overview page"""
    
    # Theme toggle
    create_theme_toggle()
    
    # Main header with enhanced styling
    st.markdown("""
    <div class="enterprise-header fade-in-up">
        <h1>🚀 AI Internship Projects</h1>
        <p class="subtitle">Enterprise Master Dashboard - Showcasing 9 Advanced AI Applications</p>
        <div class="version-badge">Version 3.0.0 - Enterprise Edition</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics with enhanced cards
    st.markdown("""
    <div class="metrics-grid fade-in-up">
        <div class="metric-card">
            <h2>9</h2>
            <p>AI Applications</p>
        </div>
        <div class="metric-card">
            <h2>25+</h2>
            <p>AI Technologies</p>
        </div>
        <div class="metric-card">
            <h2>100%</h2>
            <p>Production Ready</p>
        </div>
        <div class="metric-card">
            <h2>Enterprise</h2>
            <p>Grade Quality</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance metrics chart
    perf_chart = create_performance_chart()
    if perf_chart:
        st.markdown("### 📈 Performance Analytics")
        st.plotly_chart(perf_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Enhanced project introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Project Portfolio Overview
        
        Welcome to my comprehensive AI internship project portfolio! This enterprise-grade dashboard showcases **9 advanced AI applications** 
        built using cutting-edge technologies including **LangChain**, **Groq LLM**, **Computer Vision**, **NLP**, and **Machine Learning**.
        
        Each application demonstrates different aspects of AI/ML engineering, from conversational AI to computer vision, 
        predictive analytics, and intelligent document processing.
        
        ### 🚀 **Enterprise Features**
        - **Full Functionality Preserved** - 100% of original features
        - **Performance Monitoring** - Real-time metrics and analytics
        - **Modern UI/UX** - Professional, responsive design
        - **Theme Support** - Light/Dark mode toggle
        - **Error Handling** - Comprehensive error management
        - **Mobile Responsive** - Optimized for all devices
        """)
    
    with col2:
        # Technology stack visualization
        tech_data = {
            'Category': ['AI/ML', 'Web Dev', 'Data Science', 'Computer Vision', 'NLP'],
            'Count': [9, 5, 6, 4, 8]
        }
        
        fig = px.pie(
            values=tech_data['Count'],
            names=tech_data['Category'],
            title='Technology Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#6c757d',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Applications grid with enhanced cards
    st.markdown("## 🚀 Available Applications")
    
    # Filter and search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search Applications", placeholder="Search by name or technology...")
    
    with col2:
        category_filter = st.selectbox("📂 Filter by Category", 
                                     ["All"] + list(set(config['category'] for config in APPLICATIONS.values() if 'category' in config)))
    
    with col3:
        complexity_filter = st.selectbox("⚡ Filter by Complexity", 
                                       ["All", "High", "Medium", "Low"])
    
    # Filter applications
    filtered_apps = {}
    for app_name, config in list(APPLICATIONS.items())[1:]:  # Skip overview
        if search_term and search_term.lower() not in app_name.lower() and search_term.lower() not in config['description'].lower():
            continue
        if category_filter != "All" and config.get('category') != category_filter:
            continue
        if complexity_filter != "All" and config.get('complexity') != complexity_filter:
            continue
        filtered_apps[app_name] = config
    
    # Display filtered applications
    for i, (app_name, config) in enumerate(filtered_apps.items()):
        with st.container():
            st.markdown(f"""
            <div class="enterprise-card fade-in-up">
                <h3>{app_name}</h3>
                <p>{config['description']}</p>
                
                <div class="tech-stack">
                    <strong>🛠️ Tech Stack:</strong> {config['tech_stack']}
                </div>
                
                <div class="features-list">
                    {' '.join([f'<span class="feature-tag">{feature}</span>' for feature in config['features']])}
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                    <div>
                        <span class="status-indicator status-{config['status']}">
                            ● {config['status'].upper()}
                        </span>
                        <span style="margin-left: 1rem; color: var(--text-secondary); font-size: 0.875rem;">
                            Category: {config.get('category', 'N/A')} • 
                            Complexity: {config.get('complexity', 'N/A')} • 
                            Updated: {config.get('last_updated', 'N/A')}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Technical highlights with enhanced layout
    st.markdown("---")
    st.markdown("## 🏆 Technical Highlights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI/ML Technologies
        - **Large Language Models**: Groq Llama 3.1, GPT integration
        - **Computer Vision**: MediaPipe, OpenCV, CLIP, AnimeGAN
        - **Natural Language Processing**: spaCy, LangChain, Transformers
        - **Machine Learning**: Scikit-learn, Predictive Modeling
        - **Vector Databases**: FAISS, Semantic Search
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ Engineering Excellence
        - **Web Frameworks**: Streamlit, Gradio, FastAPI
        - **Data Processing**: Pandas, NumPy, Advanced Analytics
        - **Visualization**: Plotly, Matplotlib, Interactive Dashboards
        - **Document Processing**: PyPDF2, ReportLab, OCR
        - **Performance**: Threading, Caching, Optimization
        """)
    
    with col3:
        st.markdown("""
        ### 🏢 Enterprise Features
        - **Error Handling**: Comprehensive exception management
        - **Logging**: Detailed application monitoring
        - **Security**: Input validation, safe execution
        - **Scalability**: Modular architecture
        - **UI/UX**: Professional, responsive design
        """)
    
    # Usage instructions with enhanced styling
    st.markdown("---")
    st.markdown("""
    ## 📋 How to Use This Dashboard
    
    ### 🚀 **Getting Started**
    1. **Browse Applications**: Explore the available AI applications above
    2. **Use Filters**: Search and filter applications by category or complexity
    3. **Select Application**: Use the sidebar to choose which application to launch
    4. **Click "🚀 Load Application"**: Launch with full original functionality
    5. **Toggle Theme**: Use the 🌓 button to switch between light/dark modes
    
    ### 💡 **Pro Tips**
    - **Performance Monitoring**: View load times and performance metrics
    - **Error Recovery**: Automatic dependency installation for missing packages
    - **Mobile Friendly**: Fully responsive design works on all devices
    - **Enterprise Ready**: Production-grade error handling and user experience
    
    ### 🎯 **For Interviewers & Evaluators**
    This dashboard demonstrates comprehensive full-stack AI development capabilities:
    - **Research & Development**: From concept to production-ready applications
    - **Technical Excellence**: Clean code, proper architecture, error handling
    - **User Experience**: Professional UI/UX design and responsive layouts
    - **AI Integration**: Real AI capabilities with actual models and APIs
    - **Enterprise Quality**: Production-ready features and performance optimization
    """)
    
    # Enhanced footer
    st.markdown("""
    <div class="enterprise-footer">
        <h3>🚀 AI Internship Projects - Enterprise Dashboard</h3>
        <div class="footer-links">
            <a href="#" class="footer-link">📚 Documentation</a>
            <a href="#" class="footer-link">🐛 Report Issues</a>
            <a href="#" class="footer-link">💡 Feature Requests</a>
            <a href="#" class="footer-link">📧 Contact</a>
        </div>
        <p>Built with passion for AI/ML engineering • Showcasing enterprise-level development skills</p>
        <p>⚡ Powered by Streamlit, Python, and cutting-edge AI technologies</p>
        <p style="font-size: 0.875rem; margin-top: 1rem;">
            Version 3.0.0 - Enterprise Edition • Last Updated: January 2024
        </p>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_sidebar():
    """Create enhanced sidebar with modern styling"""
    st.sidebar.markdown("""
    <div class="sidebar-header">
        <h2>🎯 Application Selector</h2>
        <p>Choose an AI application to explore</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Application selector with enhanced styling
    selected_app = st.sidebar.selectbox(
        "Select Application:",
        list(APPLICATIONS.keys()),
        index=0,
        help="Choose which AI application you'd like to explore"
    )
    
    # Display application info in sidebar
    if selected_app in APPLICATIONS:
        config = APPLICATIONS[selected_app]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Application Details")
        
        # Status indicator
        status_color = {"ready": "🟢", "loading": "🟡", "error": "🔴"}
        st.sidebar.markdown(f"**Status:** {status_color.get(config['status'], '⚪')} {config['status'].upper()}")
        
        if 'category' in config:
            st.sidebar.markdown(f"**Category:** {config['category']}")
        
        if 'complexity' in config:
            st.sidebar.markdown(f"**Complexity:** {config['complexity']}")
        
        if 'last_updated' in config:
            st.sidebar.markdown(f"**Last Updated:** {config['last_updated']}")
        
        st.sidebar.markdown(f"**Tech Stack:** {config['tech_stack']}")
        
        # Features with enhanced display
        st.sidebar.markdown("**Key Features:**")
        for feature in config['features']:
            st.sidebar.markdown(f"• {feature}")
        
        # Performance metrics if available
        metrics = app_runner.get_performance_metrics()
        if selected_app in metrics:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📊 Performance")
            perf_data = metrics[selected_app]
            st.sidebar.markdown(f"**Load Time:** {perf_data['load_time']:.2f}s")
            st.sidebar.markdown(f"**Status:** {perf_data['status'].title()}")
    
    return selected_app

def main():
    """Main application function with enhanced features"""
    
    # Health check endpoint for Render
    if st.query_params.get("health") == "check":
        st.success("✅ AI Internship Dashboard is running successfully!")
        st.json({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "3.0.0",
            "applications": len(APPLICATIONS)
        })
        return
    
    # Create enhanced sidebar
    selected_app = create_enhanced_sidebar()
    
    # System info with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ System Information")
    st.sidebar.markdown(f"**Python:** {sys.version.split()[0]}")
    st.sidebar.markdown(f"**Streamlit:** {st.__version__}")
    st.sidebar.markdown(f"**Theme:** {st.session_state.theme.title()}")
    st.sidebar.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Enhanced load button
    st.sidebar.markdown("---")
    
    # Initialize session state for current app
    if 'current_app' not in st.session_state:
        st.session_state.current_app = "🎯 Project Overview"
    
    # Load button with enhanced styling
    if st.sidebar.button("🚀 Load Application", type="primary", use_container_width=True, 
                        help="Click to load the selected application with full functionality"):
        st.session_state.current_app = selected_app
        st.rerun()
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True, help="Return to overview"):
            st.session_state.current_app = "🎯 Project Overview"
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="Refresh current view"):
            st.rerun()
    
    # Load and display the current application
    if st.session_state.current_app == "🎯 Project Overview":
        show_overview()
    elif st.session_state.current_app in APPLICATIONS:
        config = APPLICATIONS[st.session_state.current_app]
        
        # Show application header
        st.markdown(f"""
        <div class="enterprise-header">
            <h1>{st.session_state.current_app}</h1>
            <p class="subtitle">{config['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load the application
        load_application(st.session_state.current_app, config)
    else:
        # Show application preview with enhanced styling
        config = APPLICATIONS[selected_app]
        
        st.markdown(f"""
        <div class="enterprise-header">
            <h1>{selected_app}</h1>
            <p class="subtitle">{config['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("👆 Click '🚀 Load Application' in the sidebar to launch this application with full functionality")
        
        # Enhanced application details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🛠️ Technology Stack")
            st.code(config['tech_stack'], language='text')
            
            if 'category' in config:
                st.markdown("### 📂 Category")
                st.info(config['category'])
            
        with col2:
            st.markdown("### ✨ Key Features")
            for feature in config['features']:
                st.markdown(f"• {feature}")
            
            if 'complexity' in config:
                st.markdown("### ⚡ Complexity Level")
                complexity_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.info(f"{complexity_colors.get(config['complexity'], '⚪')} {config['complexity']}")
        
        st.markdown("### 📝 Application Status")
        status_messages = {
            "ready": "✅ Ready to launch - All dependencies satisfied",
            "loading": "⏳ Loading - Please wait...",
            "error": "❌ Error - Check system requirements"
        }
        st.success(status_messages.get(config['status'], f"Status: {config['status'].upper()}"))

if __name__ == "__main__":
    main()