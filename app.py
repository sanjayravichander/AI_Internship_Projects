"""
🚀 AI INTERNSHIP PROJECTS - PUBLIC DEPLOYMENT VERSION
====================================================

This is the main entry point for the public deployment.
Optimized for Streamlit Cloud with usage management and security features.

Author: AI Intern
Version: 5.0.0 - Public Cloud Edition
"""

import streamlit as st
import sys
import os
from pathlib import Path
import traceback

# Configure page FIRST (before any other Streamlit calls)
st.set_page_config(
    page_title="🚀 AI Internship Projects - Public Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "AI Internship Projects - Public Demo v5.0.0"
    }
)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def show_welcome_message():
    """Show welcome message for public deployment."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; color: white;">
        <h1 style="margin: 0; font-size: 2.5rem;">🚀 Welcome to AI Internship Projects</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Public Demo - Showcasing 9 Advanced AI Applications
        </p>
        <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; 
                    border-radius: 20px; display: inline-block; margin-top: 1rem;">
            <small>✨ Live Demo with Usage Limits ✨</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_demo_notice():
    """Show demo limitations notice."""
    st.info("""
    📢 **Public Demo Notice**
    
    This is a **public demonstration** of the AI Internship Projects portfolio. 
    
    **Demo Features:**
    - ✅ Full access to all 9 AI applications
    - ✅ Real AI processing with usage limits
    - ✅ Professional enterprise-grade interface
    - ⚠️ Session-based usage quotas to ensure fair access
    
    **For Unlimited Access:** Deploy your own instance with personal API keys.
    """)

def show_project_overview():
    """Fallback project overview if main app fails to load."""
    st.markdown("""
    ## 🚀 AI Internship Projects Portfolio
    
    **Welcome to my comprehensive AI/ML portfolio!** This project showcases 9 advanced AI applications 
    integrated into a single enterprise-grade dashboard.
    
    ### 🤖 **Featured Applications:**
    
    1. **🎤 Voice Assistant AI** - LangChain + Groq powered conversational AI
    2. **📚 Document Intelligence Chatbot** - RAG-based document Q&A system  
    3. **🦠 COVID-19 Analytics Dashboard** - Real-time data visualization
    4. **👋 Hand Gesture Recognition** - MediaPipe computer vision
    5. **🎨 Cartoonify AI** - OpenCV image transformation
    6. **😂 Meme Classification VLM** - CLIP-based vision-language model
    7. **📊 Student Report Card Generator** - Automated PDF generation
    8. **🧠 AI Quiz Game** - Interactive educational platform
    9. **🎯 Enterprise Master Dashboard** - Unified application interface
    
    ### 💻 **Technology Stack:**
    - **AI/ML**: LangChain, Groq, Transformers, CLIP, MediaPipe
    - **Frontend**: Streamlit, Plotly, Matplotlib
    - **Backend**: Python, FastAPI, SQLite
    - **Computer Vision**: OpenCV, MediaPipe, PIL
    - **NLP**: spaCy, NLTK, Sentence Transformers
    - **Data Science**: Pandas, NumPy, Scikit-learn
    - **Cloud**: Streamlit Cloud, GitHub Actions
    
    ### 🏆 **Key Features:**
    - ✅ **Enterprise-Grade UI/UX** - Professional, responsive design
    - ✅ **Real-Time AI Processing** - Live AI model inference
    - ✅ **Usage Management** - Smart quotas and rate limiting
    - ✅ **Security** - Proper API key management and error handling
    - ✅ **Scalability** - Cloud-optimized architecture
    - ✅ **Documentation** - Comprehensive guides and examples
    
    ### 🎯 **Skills Demonstrated:**
    - **AI/ML Engineering** - Model integration, optimization, deployment
    - **Full-Stack Development** - Frontend, backend, database integration
    - **Cloud Architecture** - Scalable, secure cloud deployment
    - **DevOps** - CI/CD, environment management, monitoring
    - **User Experience** - Intuitive interfaces, error handling
    - **Software Engineering** - Clean code, documentation, testing
    
    ---
    
    ### 📞 **Contact & Links:**
    - **GitHub**: [View Source Code](https://github.com/sanjayravichander/AI_Internship_Projects)
    - **LinkedIn**: [Connect with me](https://linkedin.com/in/your-profile)
    - **Email**: sanjay.1991999@gmail.com
    
    **Thank you for exploring my AI portfolio!** 🚀
    """)
    
    # Add some metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Applications", "9", help="Complete AI applications")
    
    with col2:
        st.metric("Technologies", "15+", help="Different AI/ML technologies used")
    
    with col3:
        st.metric("Lines of Code", "10,000+", help="Total codebase size")
    
    with col4:
        st.metric("Deployment", "Cloud", help="Streamlit Cloud deployment")

def main():
    """Main application entry point with error handling."""
    try:
        # Show welcome message
        show_welcome_message()
        
        # Try to initialize environment and usage management
        try:
            from env_manager import env_manager, show_api_key_info
            from usage_manager import display_usage_info
            
            # Check API configuration
            if not show_api_key_info():
                st.stop()
            
            # Display usage information in sidebar
            display_usage_info()
            
            # Show deployment status if in debug mode
            env_manager.display_deployment_status()
            
        except ImportError as import_err:
            st.warning(f"⚠️ Some management features unavailable: {import_err}")
            st.info("🎯 Running in basic mode - all core features still work!")
        
        # Show demo notice
        show_demo_notice()
        
        # Import and run the main application
        try:
            from master_app_enterprise import main as enterprise_main
            enterprise_main()
        except ImportError:
            # Fallback: show a basic version
            st.error("❌ Main application not found. Showing project overview instead.")
            show_project_overview()
        
    except ImportError as e:
        st.error(f"""
        ## 🚨 Import Error
        
        There was an issue importing the main application: {e}
        
        **Common Solutions:**
        1. Ensure all dependencies are installed:
        ```bash
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
        ```
        
        2. Check that all required files are present
        
        3. Verify Python path configuration
        
        **For Streamlit Cloud:** Ensure all files are committed to your repository.
        """)
        
        # Show detailed error in debug mode
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            st.code(traceback.format_exc())
            
    except Exception as e:
        st.error(f"""
        ## 🚨 Application Error
        
        An unexpected error occurred: {e}
        
        **Troubleshooting Steps:**
        1. Refresh the page
        2. Clear browser cache
        3. Check your internet connection
        4. Try again in a few minutes
        
        If the problem persists, this may be a temporary service issue.
        """)
        
        # Show detailed error in debug mode
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            st.code(traceback.format_exc())
        
        # Provide fallback information
        st.markdown("---")
        st.markdown("""
        ### 📋 Project Information
        
        **AI Internship Projects Portfolio**
        - 🤖 Voice Assistant AI with LangChain & Groq
        - 📚 Document Intelligence Chatbot
        - 🦠 COVID-19 Analytics Dashboard  
        - 👋 Hand Gesture Recognition
        - 🎨 Cartoonify AI
        - 😂 Meme Classification VLM
        - 📊 Student Report Card Generator
        - 🧠 AI Quiz Game
        - 🎯 Enterprise Master Dashboard
        
        **Technology Stack:** Python, Streamlit, LangChain, Groq, OpenCV, MediaPipe, CLIP, and more.
        """)

if __name__ == "__main__":
    main()