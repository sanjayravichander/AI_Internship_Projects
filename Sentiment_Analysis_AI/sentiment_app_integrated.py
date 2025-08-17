"""
💭 SENTIMENT ANALYSIS AI - INTEGRATED APPLICATION
================================================

Advanced sentiment analysis system using fine-tuned transformer models
with real-time text processing, confidence scoring, and interactive visualizations.

Features:
- Real-time sentiment analysis
- Confidence scoring with animated visualizations
- Support for fine-tuned and pre-trained models
- Interactive UI with flip animations
- Multiple sentiment classes (positive, negative, neutral)

Author: AI Intern
Version: 2.0.0 - Integrated Edition
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import traceback

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import required libraries
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"❌ Missing required dependencies: {e}")
    st.info("💡 Please install the required packages: `pip install transformers torch numpy pandas`")
    st.stop()

# Page configuration (commented out for master app integration)
# st.set_page_config(
#     page_title="💭 AI Sentiment Analyzer",
#     layout="centered",
#     page_icon="💭"
# )

# Enhanced CSS with modern styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .sentiment-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Header Styles */
    .sentiment-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .sentiment-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sentiment-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Flip Animation Styles */
    .flip-box {
        background-color: transparent;
        width: 100%;
        max-width: 400px;
        height: 120px;
        perspective: 1000px;
        margin: 2rem auto;
    }
    
    .flip-box-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
    }
    
    .flip-box:hover .flip-box-inner {
        transform: rotateY(180deg);
    }
    
    .flip-box-front, .flip-box-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.8rem;
        font-weight: 600;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .flip-box-front {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
    }
    
    .flip-box-back {
        transform: rotateY(180deg);
        color: white;
        font-size: 1.4rem;
    }
    
    /* Progress Bar Styles */
    .confidence-container {
        margin: 2rem 0;
        padding: 1.5rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
    }
    
    .confidence-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .bar-container {
        width: 100%;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 25px;
        overflow: hidden;
        height: 40px;
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .bar {
        height: 100%;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 25px;
        position: relative;
        overflow: hidden;
    }
    
    .bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .confidence-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-weight: 600;
        font-size: 1rem;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        z-index: 10;
    }
    
    /* Input Styles */
    .stTextArea > div > div > textarea {
        border-radius: 16px;
        border: 2px solid #e9ecef;
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Status Cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .status-loading {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        color: #2d3436;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2d3436;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff7675 0%, #fd79a8 100%);
        color: white;
    }
    
    /* Metrics Display */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .sentiment-header h1 {
            font-size: 2rem;
        }
        
        .flip-box {
            height: 100px;
        }
        
        .flip-box-front, .flip-box-back {
            font-size: 1.4rem;
        }
        
        .metrics-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Cached Model Loading ----------------------
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model with Streamlit caching to avoid re-downloading"""
    try:
        # Try to load fine-tuned model first
        model_path = Path(__file__).parent / "finetune" / "files" / "backend" / "model"
        
        if model_path.exists() and any(model_path.iterdir()):
            st.info("🔄 Loading fine-tuned model...")
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            model_info = {
                "type": "Fine-tuned Model",
                "path": str(model_path),
                "status": "loaded"
            }
            st.success("✅ Fine-tuned model loaded successfully!")
        else:
            # Fallback to pre-trained model
            st.info("🔄 Loading pre-trained model...")
            default_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            # Force PyTorch backend to avoid TensorFlow issues
            model = AutoModelForSequenceClassification.from_pretrained(default_model)
            tokenizer = AutoTokenizer.from_pretrained(default_model)
            sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework="pt")
            model_info = {
                "type": "Pre-trained Model",
                "model": default_model,
                "status": "loaded"
            }
            st.success("✅ Pre-trained model loaded successfully!")
            
        return sentiment_pipeline, model_info
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        model_info = {
            "type": "Error",
            "error": str(e),
            "status": "failed"
        }
        raise e

class SentimentAnalyzer:
    """Advanced sentiment analysis with model management"""
    
    def __init__(self):
        self.pipeline, self.model_info = load_sentiment_model()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of input text"""
        if not self.pipeline:
            raise ValueError("Model not loaded")
        
        try:
            result = self.pipeline(text)[0]
            return {
                "label": result["label"].lower(),
                "score": round(result["score"], 4),
                "confidence": round(result["score"] * 100, 2)
            }
        except Exception as e:
            st.error(f"❌ Error analyzing sentiment: {str(e)}")
            return None

def create_sentiment_visualization(label, score, confidence):
    """Create animated sentiment visualization"""
    
    # Emoji and color mapping
    emoji_map = {
        "positive": "😄",
        "negative": "😠", 
        "neutral": "😐",
        "label_0": "😠",  # Negative for some models
        "label_1": "😐",  # Neutral for some models  
        "label_2": "😄",  # Positive for some models
    }
    
    color_map = {
        "positive": "#2ecc71",  # Green
        "negative": "#e74c3c",  # Red
        "neutral": "#f39c12",   # Orange
        "label_0": "#e74c3c",   # Red (Negative)
        "label_1": "#f39c12",   # Orange (Neutral)
        "label_2": "#2ecc71",   # Green (Positive)
    }
    
    # Get emoji and color
    emoji = emoji_map.get(label, "🤔")
    color = color_map.get(label, "#95a5a6")
    
    # Normalize label for display
    display_label = label.replace("label_0", "negative").replace("label_1", "neutral").replace("label_2", "positive")
    
    # Create flip animation
    st.markdown(f"""
    <div class="flip-box">
        <div class="flip-box-inner">
            <div class="flip-box-front">
                {emoji} {display_label.capitalize()}
            </div>
            <div class="flip-box-back" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                Confidence: {confidence}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create confidence bar
    st.markdown(f"""
    <div class="confidence-container">
        <div class="confidence-label">Confidence Level</div>
        <div class="bar-container">
            <div class="bar" style="width: {confidence}%; background: linear-gradient(90deg, {color} 0%, {color}cc 100%);">
            </div>
            <div class="confidence-text">{confidence}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_model_info(analyzer):
    """Display model information and metrics"""
    st.markdown("### 🤖 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{analyzer.model_info.get('type', 'Unknown')}</div>
            <div class="metric-label">Model Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = analyzer.model_info.get('status', 'unknown')
        status_emoji = {"loaded": "✅", "failed": "❌", "unknown": "❓"}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status_emoji.get(status, '❓')}</div>
            <div class="metric-label">Status: {status.title()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        model_name = analyzer.model_info.get('model', 'Custom Model')
        if len(model_name) > 20:
            model_name = model_name.split('/')[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">🧠</div>
            <div class="metric-label">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="sentiment-header">
        <h1>💭 AI Sentiment Analyzer</h1>
        <p>Advanced sentiment analysis with real-time processing and confidence scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        with st.spinner("🔄 Initializing sentiment analysis model..."):
            try:
                st.session_state.analyzer = SentimentAnalyzer()
            except Exception as e:
                st.error("❌ Failed to initialize model. Please check your setup.")
                st.exception(e)
                return
    
    analyzer = st.session_state.analyzer
    
    # Show model information
    show_model_info(analyzer)
    
    st.markdown("---")
    
    # Input section
    st.markdown("### 📝 Text Analysis")
    text = st.text_area(
        "Enter your text for sentiment analysis:",
        height=150,
        placeholder="e.g., I love this new AI technology! It's amazing how accurate it is.",
        help="Type or paste any text to analyze its sentiment"
    )
    
    # Analysis section
    if text.strip():
        try:
            with st.spinner("🔍 Analyzing sentiment..."):
                result = analyzer.analyze_sentiment(text)
            
            if result:
                # Create visualization
                create_sentiment_visualization(
                    result['label'], 
                    result['score'], 
                    result['confidence']
                )
                
                # Show detailed results
                st.markdown("### 📊 Detailed Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['label'].title()}</div>
                        <div class="metric-label">Predicted Sentiment</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['score']}</div>
                        <div class="metric-label">Raw Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result['confidence']}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                if result['confidence'] >= 80:
                    st.success(f"🎯 **High Confidence**: The model is very confident that this text expresses **{result['label']}** sentiment.")
                elif result['confidence'] >= 60:
                    st.info(f"📊 **Medium Confidence**: The model believes this text is likely **{result['label']}** with moderate confidence.")
                else:
                    st.warning(f"🤔 **Low Confidence**: The model suggests **{result['label']}** sentiment but with low confidence. The text might be ambiguous.")
                
        except Exception as e:
            st.error("❌ Error analyzing text. Please try again.")
            with st.expander("🔍 Error Details"):
                st.exception(e)
    else:
        # Show example when no input
        st.markdown("""
        <div class="status-card status-ready">
            <h3>🚀 Ready for Analysis</h3>
            <p>Enter some text above to see real-time sentiment analysis with confidence scoring and animated visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show examples
        st.markdown("### 💡 Try These Examples")
        
        examples = [
            "I absolutely love this new product! It's fantastic!",
            "This is the worst experience I've ever had.",
            "The weather is okay today, nothing special.",
            "I'm feeling quite optimistic about the future.",
            "This movie was disappointing and boring."
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(f"Example {i+1}", key=f"example_{i}", help=example):
                    st.session_state.example_text = example
                    st.rerun()
        
        # Display selected example
        if hasattr(st.session_state, 'example_text'):
            st.text_area("Selected Example:", value=st.session_state.example_text, key="example_display")
    
    # Footer with technical details
    st.markdown("---")
    st.markdown("### 🛠️ Technical Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 Model Architecture:**
        - Transformer-based sentiment classification
        - Fine-tuned RoBERTa or similar architecture
        - Multi-class sentiment prediction
        - Confidence scoring with softmax probabilities
        """)
    
    with col2:
        st.markdown("""
        **⚡ Features:**
        - Real-time text processing
        - Interactive animated visualizations
        - Confidence-based interpretations
        - Support for custom fine-tuned models
        """)

if __name__ == "__main__":
    main()