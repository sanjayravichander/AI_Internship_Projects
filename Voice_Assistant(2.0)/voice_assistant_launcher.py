"""
🎤 Voice Assistant 2.0 - LiveKit Playground Launcher
==================================================

Simple Streamlit UI that connects users to the LiveKit playground
for the advanced Alice AI Assistant with voice capabilities.

Author: AI Intern
Version: 2.0.0 - LiveKit Integration
"""

import streamlit as st
import webbrowser
import os
from datetime import datetime
import json

# Import configuration
try:
    from config import (
        LIVEKIT_PLAYGROUND_URL, ASSISTANT_NAME, ASSISTANT_VERSION,
        ASSISTANT_DESCRIPTION, THEME_COLOR, SECONDARY_COLOR, STATUS, LAST_UPDATED
    )
except ImportError:
    # Fallback configuration if config.py is not available
    LIVEKIT_PLAYGROUND_URL = "https://agents-playground.livekit.io/"
    ASSISTANT_NAME = "Alice"
    ASSISTANT_VERSION = "2.0.0"
    ASSISTANT_DESCRIPTION = "Advanced AI Assistant with LiveKit Integration"
    THEME_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    STATUS = "ready"
    LAST_UPDATED = "2024-01-18"

# Configure page
st.set_page_config(
    page_title="🎤 Voice Assistant 2.0 - Alice",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Voice Assistant theme
st.markdown(f"""
<style>
    .voice-header {{
        background: linear-gradient(135deg, {THEME_COLOR} 0%, {SECONDARY_COLOR} 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }}
    
    .voice-header h1 {{
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .voice-header p {{
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }}
    
    .feature-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {THEME_COLOR};
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }}
    
    .feature-card h3 {{
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }}
    
    .feature-card p {{
        color: #6c757d;
        margin: 0;
    }}
    
    .playground-button {{
        background: linear-gradient(135deg, {THEME_COLOR} 0%, {SECONDARY_COLOR} 100%);
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }}
    
    .playground-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }}
    
    .status-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .status-ready {{
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    
    .tech-stack {{
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        color: #2c3e50;
    }}
    
    .info-box {{
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    
    .warning-box {{
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown(f"""
    <div class="voice-header">
        <h1>🎤 {ASSISTANT_NAME} - Voice Assistant {ASSISTANT_VERSION}</h1>
        <p>{ASSISTANT_DESCRIPTION}</p>
        <span class="status-badge status-ready">● {STATUS.title()}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"## 🚀 Welcome to {ASSISTANT_NAME} Voice Assistant")
        
        st.markdown(f"""
        {ASSISTANT_NAME} is an advanced AI assistant powered by Google's latest LLM technology and LiveKit's 
        real-time communication platform. Experience natural voice conversations with intelligent 
        system control capabilities.
        """)
        
        # Features section
        st.markdown("### ✨ Key Features")
        
        features = [
            {
                "icon": "🗣️",
                "title": "Natural Voice Conversation",
                "description": f"Engage in fluid, natural conversations with {ASSISTANT_NAME} using advanced speech recognition and synthesis"
            },
            {
                "icon": "🌐",
                "title": "Web Integration",
                "description": "Search the web, get weather updates, and access real-time information"
            },
            {
                "icon": "💻",
                "title": "System Control",
                "description": "Open applications, take screenshots, and execute system commands"
            },
            {
                "icon": "📧",
                "title": "Communication Hub",
                "description": "Send emails, WhatsApp messages, and manage your communications"
            },
            {
                "icon": "🎵",
                "title": "Media Control",
                "description": "Play YouTube videos, control media, and manage entertainment"
            },
            {
                "icon": "⚡",
                "title": "Fast Startup",
                "description": "Optimized LiveKit integration with quick connection and response times"
            }
        ]
        
        for feature in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Tech Stack
        st.markdown("""
        <div class="tech-stack">
            <h4>🛠️ Technology Stack</h4>
            <ul>
                <li><strong>LiveKit Agents</strong> - Real-time communication</li>
                <li><strong>Google LLM & TTS</strong> - Advanced AI capabilities</li>
                <li><strong>System Integration</strong> - Native OS control</li>
                <li><strong>Web APIs</strong> - External service integration</li>
                <li><strong>Fast Startup</strong> - Optimized performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Status Information
        st.markdown(f"""
        <div class="info-box">
            <h4>📊 System Status</h4>
            <p><strong>Status:</strong> <span class="status-badge status-ready">{STATUS.title()}</span></p>
            <p><strong>Version:</strong> {ASSISTANT_VERSION}</p>
            <p><strong>Last Updated:</strong> {LAST_UPDATED}</p>
            <p><strong>Platform:</strong> LiveKit Playground</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown(f"""
        <div class="warning-box">
            <h4>📋 Instructions</h4>
            <p>1. Click the "Launch LiveKit Playground" button below</p>
            <p>2. Allow microphone access when prompted</p>
            <p>3. Wait for {ASSISTANT_NAME} to initialize</p>
            <p>4. Start speaking to begin your conversation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Playground Launch Section
    st.markdown("---")
    st.markdown("## 🎯 Launch Voice Assistant")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h3>Ready to talk with {ASSISTANT_NAME}?</h3>
            <p>Click the button below to open the LiveKit playground and start your voice conversation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Playground URL from configuration
        playground_url = LIVEKIT_PLAYGROUND_URL
        
        if st.button("🚀 Launch LiveKit Playground", key="launch_playground", help="Open LiveKit playground in a new tab"):
            st.markdown(f"""
            <script>
                window.open('{playground_url}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            
            st.success("🎉 **LiveKit Playground launched!** Check your browser for a new tab.")
            st.info("💡 **Tip:** Make sure to allow microphone access for the best experience.")
        
        # Alternative manual link
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1rem;">
            <p>Or manually visit: <a href="{playground_url}" target="_blank">{playground_url}</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Information
    st.markdown("---")
    st.markdown("## 📚 Additional Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Configuration
        - **Agent File:** `agent.py`
        - **Tools:** `tools_simple.py`
        - **Prompts:** `prompts_2.py`
        - **Environment:** Requires `.env` file with API keys
        """)
        
        st.markdown("""
        ### 🎯 Use Cases
        - Personal productivity assistant
        - System automation
        - Information retrieval
        - Communication management
        - Media control
        """)
    
    with col2:
        st.markdown("""
        ### 🔑 Required API Keys
        - `GOOGLE_API_KEY` - For LLM and TTS
        - `LIVEKIT_URL` - LiveKit server URL
        - `LIVEKIT_API_KEY` - LiveKit API key
        - `LIVEKIT_API_SECRET` - LiveKit API secret
        """)
        
        st.markdown("""
        ### 🆘 Troubleshooting
        - Ensure microphone permissions are granted
        - Check internet connectivity
        - Verify API keys are configured
        - Try refreshing the playground page
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6c757d; padding: 1rem;">
        <p>🤖 {ASSISTANT_NAME} Voice Assistant {ASSISTANT_VERSION} | Powered by LiveKit & Google AI</p>
        <p>Built with ❤️ for the AI Internship Projects</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()