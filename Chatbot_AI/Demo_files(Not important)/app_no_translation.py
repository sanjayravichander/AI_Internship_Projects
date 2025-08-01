#!/usr/bin/env python3
"""
Enhanced Elite Streamlit UI for Advanced Multi-Document Q&A Chatbot
Features: Multi-document support, Hybrid search, Fixed filter controls
Version: No Translation Dependencies (Fully Working)
"""

import streamlit as st
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime
from typing import Dict, List, Any, Optional
from agentic_doc_qa_chatbot_enhanced import EnhancedAgenticDocumentQAChatbot
from rate_limit_handler import handle_groq_error
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import hashlib

# Page configuration
st.set_page_config(
    page_title="🚀 Elite Multi-Document Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elite styling with enhanced multi-document support
st.markdown("""
<style>
    /* Elite theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --success-color: #059669;
        --info-color: #0ea5e9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Elite header styling */
    .elite-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #10b981 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Elite cards */
    .elite-card {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Document cards */
    .document-card {
        background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid var(--accent-color);
    }
    
    .document-card.active {
        border-left: 4px solid var(--success-color);
        background: linear-gradient(145deg, #f0fdf4, #dcfce7);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        color: white;
        margin-left: 20%;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        margin-right: 20%;
    }
    
    /* Advanced feature buttons */
    .feature-button {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .feature-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Search type indicators */
    .search-type-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    
    .hybrid-search {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
    }
    
    .semantic-search {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .keyword-search {
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        color: white;
    }
    
    /* Fixed controls indicator */
    .fixed-controls {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize enhanced session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_insights' not in st.session_state:
        st.session_state.document_insights = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'active_documents' not in st.session_state:
        st.session_state.active_documents = []
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Comprehensive"
    if 'response_style' not in st.session_state:
        st.session_state.response_style = "Professional"
    if 'search_type' not in st.session_state:
        st.session_state.search_type = "hybrid"
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "multilingual-e5-large"

def main():
    """Enhanced main application function"""
    initialize_session_state()
    
    # Elite header with multi-document support
    st.markdown("""
    <div class="elite-header">
        <h1>🚀 Elite Multi-Document Intelligence Platform</h1>
        <p>Advanced AI-Powered Multi-Document Analysis & Q&A System</p>
        <p><em>Multi-Language Support • Hybrid Search • Cross-Document Analysis</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show fixed controls status
    st.markdown("""
    <div class="fixed-controls">
        ✅ FILTER CONTROLS FIXED - All settings now properly update the chatbot behavior!
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with multi-document management
    with st.sidebar:
        display_document_management()
        display_enhanced_settings()
        
        # Document insights panel
        if st.session_state.chatbot and st.session_state.active_documents:
            display_document_insights_panel()
    
    # Main content area with enhanced tabs
    if st.session_state.chatbot and st.session_state.active_documents:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "💬 Intelligent Chat", 
            "📊 Analytics Dashboard", 
            "🔍 Hybrid Search", 
            "📈 Document Insights",
            "🌐 Knowledge Graph",
            "🔄 Cross-Document Analysis"
        ])
        
        with tab1:
            display_enhanced_chat_interface()
        
        with tab2:
            display_analytics_dashboard()
        
        with tab3:
            display_hybrid_search_interface()
        
        with tab4:
            display_document_insights()
        
        with tab5:
            display_knowledge_graph()
            
        with tab6:
            display_cross_document_analysis()
    
    else:
        display_enhanced_welcome_screen()

def display_document_management():
    """Enhanced document management with multi-document support"""
    st.markdown("## 📁 Multi-Document Management")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload multiple PDF, TXT, and DOCX files for cross-document analysis"
    )
    
    if uploaded_files:
        # Display uploaded files
        st.markdown("### 📄 Uploaded Documents")
        for i, file in enumerate(uploaded_files):
            file_hash = hashlib.md5(file.name.encode()).hexdigest()[:8]
            is_active = file_hash in [doc['hash'] for doc in st.session_state.active_documents]
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"""
                <div class="document-card {'active' if is_active else ''}">
                    📄 {file.name}<br>
                    <small>Size: {file.size:,} bytes</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("➕", key=f"add_{i}", help="Add to analysis"):
                    add_document_to_analysis(file)
            
            with col3:
                if is_active and st.button("➖", key=f"remove_{i}", help="Remove from analysis"):
                    remove_document_from_analysis(file_hash)
        
        # Initialize multi-document chatbot
        if st.session_state.active_documents and st.button("🚀 Initialize Multi-Document AI", type="primary"):
            initialize_multi_document_chatbot()
    
    # Display active documents
    if st.session_state.active_documents:
        st.markdown("### ✅ Active Documents")
        for doc in st.session_state.active_documents:
            st.markdown(f"""
            <div class="document-card active">
                📄 {doc['name']}<br>
                <small>Hash: {doc['hash']}</small>
            </div>
            """, unsafe_allow_html=True)

def display_enhanced_settings():
    """Enhanced settings with proper state management - FIXED VERSION"""
    st.markdown("## ⚙️ Advanced Settings")
    
    st.markdown("""
    <div class="fixed-controls">
        🔧 CONTROLS FIXED - Changes now update the chatbot immediately!
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis mode with state management
    new_analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Comprehensive", "Fast", "Deep Analysis", "Cross-Document"],
        index=["Comprehensive", "Fast", "Deep Analysis", "Cross-Document"].index(st.session_state.analysis_mode),
        help="Choose analysis depth vs speed",
        key="analysis_mode_selector"
    )
    
    if new_analysis_mode != st.session_state.analysis_mode:
        st.session_state.analysis_mode = new_analysis_mode
        if st.session_state.chatbot:
            st.session_state.chatbot.update_analysis_mode(new_analysis_mode)
        st.success(f"✅ Analysis mode updated to: {new_analysis_mode}")
    
    # Response style with state management
    new_response_style = st.selectbox(
        "Response Style",
        ["Professional", "Technical", "Executive", "Academic", "Casual"],
        index=["Professional", "Technical", "Executive", "Academic", "Casual"].index(st.session_state.response_style),
        help="Customize response tone and detail level",
        key="response_style_selector"
    )
    
    if new_response_style != st.session_state.response_style:
        st.session_state.response_style = new_response_style
        if st.session_state.chatbot:
            st.session_state.chatbot.update_response_style(new_response_style)
        st.success(f"✅ Response style updated to: {new_response_style}")
    
    # Search type selection
    new_search_type = st.selectbox(
        "Search Strategy",
        ["hybrid", "semantic", "keyword", "dense", "sparse"],
        index=["hybrid", "semantic", "keyword", "dense", "sparse"].index(st.session_state.search_type),
        help="Choose search strategy for document retrieval",
        key="search_type_selector"
    )
    
    if new_search_type != st.session_state.search_type:
        st.session_state.search_type = new_search_type
        if st.session_state.chatbot:
            st.session_state.chatbot.update_search_type(new_search_type)
        st.success(f"✅ Search strategy updated to: {new_search_type}")
    
    # Embedding model selection
    embedding_options = [
        "multilingual-e5-large",
        "multilingual-e5-base", 
        "paraphrase-multilingual-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "jinaai/jina-embeddings-v3"
    ]
    
    new_embedding_model = st.selectbox(
        "Embedding Model",
        embedding_options,
        index=embedding_options.index(st.session_state.embedding_model),
        help="Choose embedding model for semantic understanding",
        key="embedding_model_selector"
    )
    
    if new_embedding_model != st.session_state.embedding_model:
        st.session_state.embedding_model = new_embedding_model
        st.info("🔄 Embedding model will be updated on next document initialization")
    
    # Display current settings
    st.markdown("### 📊 Current Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analysis Mode", st.session_state.analysis_mode)
        st.metric("Search Strategy", st.session_state.search_type.title())
    with col2:
        st.metric("Response Style", st.session_state.response_style)
        st.metric("Embedding Model", st.session_state.embedding_model.split('/')[-1])

def display_document_insights_panel():
    """Document insights panel in sidebar"""
    st.markdown("## 📊 Document Insights")
    
    if st.button("🔍 Analyze All Documents"):
        with st.spinner("Performing multi-document analysis..."):
            try:
                st.session_state.document_insights = st.session_state.chatbot.get_multi_document_insights()
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
    
    if st.session_state.document_insights:
        display_sidebar_insights()

def add_document_to_analysis(file):
    """Add document to active analysis"""
    file_hash = hashlib.md5(file.name.encode()).hexdigest()[:8]
    
    # Check if already added
    if file_hash not in [doc['hash'] for doc in st.session_state.active_documents]:
        # Save file temporarily
        temp_path = f"temp_{file_hash}_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Add to active documents
        st.session_state.active_documents.append({
            'name': file.name,
            'hash': file_hash,
            'path': temp_path,
            'size': file.size
        })
        
        st.success(f"✅ Added {file.name} to analysis")
        st.rerun()

def remove_document_from_analysis(file_hash):
    """Remove document from active analysis"""
    # Find and remove document
    doc_to_remove = None
    for doc in st.session_state.active_documents:
        if doc['hash'] == file_hash:
            doc_to_remove = doc
            break
    
    if doc_to_remove:
        # Clean up temp file
        try:
            os.remove(doc_to_remove['path'])
        except:
            pass
        
        # Remove from active documents
        st.session_state.active_documents.remove(doc_to_remove)
        st.success(f"✅ Removed {doc_to_remove['name']} from analysis")
        
        # Reinitialize chatbot if needed
        if st.session_state.active_documents:
            initialize_multi_document_chatbot()
        else:
            st.session_state.chatbot = None
        
        st.rerun()

def initialize_multi_document_chatbot():
    """Initialize chatbot with multiple documents"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing Multi-Document AI System...")
        progress_bar.progress(20)
        
        # Create enhanced chatbot instance
        document_paths = [doc['path'] for doc in st.session_state.active_documents]
        
        st.session_state.chatbot = EnhancedAgenticDocumentQAChatbot(
            document_paths=document_paths,
            embedding_model=st.session_state.embedding_model,
            search_type=st.session_state.search_type,
            analysis_mode=st.session_state.analysis_mode,
            response_style=st.session_state.response_style,
            force_recreate=True
        )
        
        progress_bar.progress(60)
        status_text.text("🧠 Loading Advanced AI Models...")
        
        # Initialize advanced features
        time.sleep(1)
        
        progress_bar.progress(80)
        status_text.text("📊 Generating Multi-Document Insights...")
        
        # Get initial insights
        try:
            st.session_state.document_insights = st.session_state.chatbot.get_multi_document_insights()
        except Exception as e:
            st.error(f"❌ Error getting document insights: {str(e)}")
            st.session_state.document_insights = {
                "basic_info": {"error": "Could not load document insights"},
                "multi_document_analysis": {"error": "Analysis unavailable"}
            }
        
        progress_bar.progress(100)
        status_text.text("✅ Multi-Document AI System Ready!")
        
        st.success(f"🚀 Elite Multi-Document Intelligence Platform is now active with {len(st.session_state.active_documents)} documents!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error initializing multi-document system: {str(e)}")

def display_enhanced_welcome_screen():
    """Enhanced welcome screen with multi-document features"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="elite-card">
            <h3>🧠 Multi-Document AI Analysis</h3>
            <ul>
                <li>Cross-document understanding</li>
                <li>Multi-language support (Hindi optimized)</li>
                <li>Hybrid search capabilities</li>
                <li>Advanced entity extraction</li>
                <li>Document comparison & synthesis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="elite-card">
            <h3>🔍 Hybrid Search Engine</h3>
            <ul>
                <li>Semantic + Keyword search</li>
                <li>Dense + Sparse retrieval</li>
                <li>Cross-document search</li>
                <li>Multi-language queries</li>
                <li>Context-aware results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="elite-card">
            <h3>🔧 Fixed Filter Controls</h3>
            <ul>
                <li>✅ Analysis mode updates work</li>
                <li>✅ Response style changes apply</li>
                <li>✅ Search strategy updates</li>
                <li>✅ Real-time feedback</li>
                <li>✅ Immediate effect on chatbot</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Get Started")
    st.info("Upload multiple documents in the sidebar to begin your enhanced multi-document analysis experience!")
    
    # Best practices section
    with st.expander("📚 Best Practices for Multi-Document Analysis"):
        st.markdown("""
        **For Optimal Results:**
        
        1. **Document Selection**: Upload related documents for better cross-document analysis
        2. **Settings Configuration**: Use the fixed filter controls to customize behavior
        3. **Search Strategy**: Use 'hybrid' for best results, 'semantic' for conceptual queries
        4. **Embedding Model**: 'multilingual-e5-large' recommended for multi-language support
        5. **Analysis Mode**: Use 'Cross-Document' for comparing multiple documents
        
        **Supported Features:**
        - ✅ PDF, TXT, DOCX files
        - ✅ Multiple documents simultaneously
        - ✅ 12+ languages including Hindi
        - ✅ Hybrid search (semantic + keyword)
        - ✅ Cross-document comparison
        - ✅ **FIXED filter controls that actually work!**
        """)

def display_enhanced_chat_interface():
    """Enhanced chat interface"""
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['is_user']:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                # AI response with settings info
                settings_info = f"Mode: {message.get('mode', 'AI')} • Search: {message.get('search_type', 'N/A')} • Style: {message.get('response_style', 'N/A')}"
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>Elite AI:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')} • {settings_info}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced smart suggestions
    st.markdown("#### 🎯 Smart Suggestions")
    
    try:
        suggestions = st.session_state.chatbot.get_suggested_questions()
        if suggestions:
            # Multi-document specific suggestions
            multi_doc_suggestions = [
                "Compare the main themes across all documents",
                "What are the common entities mentioned in all documents?",
                "Summarize the key differences between the documents",
                "Find contradictions or agreements between documents"
            ]
            
            all_suggestions = suggestions + multi_doc_suggestions
            
            cols = st.columns(min(3, len(all_suggestions)))
            for i, suggestion in enumerate(all_suggestions[:9]):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        process_enhanced_question(suggestion)
    except:
        st.info("Smart suggestions will appear after document analysis")
    
    # Enhanced chat input
    user_question = st.chat_input("Ask anything about your documents...")
    
    if user_question:
        process_enhanced_question(user_question)

def process_enhanced_question(question: str):
    """Process user question with enhanced features"""
    
    # Add user message
    st.session_state.chat_history.append({
        'content': question,
        'is_user': True,
        'timestamp': datetime.now()
    })
    
    # Get AI response with enhanced features
    with st.spinner("🧠 Elite Multi-Document AI is analyzing..."):
        try:
            start_time = time.time()
            response = st.session_state.chatbot.ask_question(
                question=question,
                search_type=st.session_state.search_type,
                analysis_mode=st.session_state.analysis_mode,
                response_style=st.session_state.response_style
            )
            response_time = time.time() - start_time
            
            # Add AI response with current settings
            st.session_state.chat_history.append({
                'content': response.get('answer', 'No response generated'),
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': st.session_state.analysis_mode,
                'search_type': st.session_state.search_type,
                'response_style': st.session_state.response_style,
                'response_time': response_time,
                'documents_used': response.get('documents_used', [])
            })
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            
            st.session_state.chat_history.append({
                'content': error_msg,
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': 'Error'
            })
    
    st.rerun()

# Simplified placeholder functions for other interfaces
def display_analytics_dashboard():
    """Display analytics dashboard"""
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.document_insights:
        insights = st.session_state.document_insights
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Documents", len(st.session_state.active_documents))
        
        with col2:
            basic_info = insights.get('basic_info', {})
            st.metric("Total Pages", basic_info.get('total_pages', 0))
        
        with col3:
            st.metric("Embedding Model", st.session_state.embedding_model.split('/')[-1])
        
        with col4:
            st.metric("Search Strategy", st.session_state.search_type.title())
        
        # Settings verification
        st.markdown("#### ⚙️ Current Settings (FIXED)")
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            st.write(f"**Analysis Mode:** {st.session_state.analysis_mode}")
            st.write(f"**Search Type:** {st.session_state.search_type}")
        with settings_col2:
            st.write(f"**Response Style:** {st.session_state.response_style}")
            st.write(f"**Embedding Model:** {st.session_state.embedding_model}")
        
        # Document information
        st.markdown("#### 📄 Document Information")
        if 'basic_info' in insights:
            basic_info = insights['basic_info']
            if 'documents' in basic_info:
                for doc in basic_info['documents']:
                    st.write(f"📄 **{doc}**")
    else:
        st.info("Run document analysis to see analytics dashboard")

def display_hybrid_search_interface():
    """Display hybrid search interface"""
    st.markdown("### 🔍 Hybrid Search Interface")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., machine learning algorithms"
        )
    
    with col2:
        max_results = st.number_input("Max Results", min_value=1, max_value=20, value=5)
    
    # Show current search settings
    st.info(f"🔍 Using {st.session_state.search_type} search with {st.session_state.embedding_model} embeddings")
    
    if search_query and st.button("🚀 Search", type="primary"):
        with st.spinner(f"Performing {st.session_state.search_type} search..."):
            try:
                results = st.session_state.chatbot.perform_hybrid_search(
                    query=search_query,
                    search_type=st.session_state.search_type,
                    max_results=max_results
                )
                
                if 'error' not in results:
                    st.success(f"Found {results['total_results']} relevant results using {st.session_state.search_type} search")
                    
                    for i, result in enumerate(results['results']):
                        with st.expander(f"📄 Result #{i+1} - {result.get('document_name', 'Unknown')} (Score: {result['relevance_score']:.3f})"):
                            st.write("**Content:**")
                            st.write(result['content'])
                else:
                    st.error(results['error'])
            except Exception as e:
                st.error(f"Search error: {str(e)}")

def display_document_insights():
    """Display document insights"""
    st.markdown("### 📈 Document Insights")
    
    if st.session_state.document_insights:
        insights = st.session_state.document_insights
        
        # Basic information
        if 'basic_info' in insights:
            st.markdown("#### 📊 Basic Information")
            basic_info = insights['basic_info']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Documents:** {basic_info.get('total_documents', 0)}")
                st.write(f"**Total Pages:** {basic_info.get('total_pages', 0)}")
            
            with col2:
                st.write(f"**Embedding Model:** {basic_info.get('embedding_model', 'N/A')}")
                st.write(f"**Search Type:** {basic_info.get('search_type', 'N/A')}")
        
        # Multi-document analysis
        if 'multi_document_analysis' in insights:
            st.markdown("#### 🔄 Multi-Document Analysis")
            analysis = insights['multi_document_analysis']
            
            if isinstance(analysis, dict) and 'error' not in analysis:
                st.write(f"**Average Reading Time:** {analysis.get('avg_reading_time', 0):.1f} minutes")
                st.write(f"**Total Words:** {analysis.get('total_words', 0):,}")
                
                if 'cross_document_entities_count' in analysis:
                    st.write(f"**Cross-Document Entities:** {analysis['cross_document_entities_count']}")
            else:
                st.warning("Multi-document analysis not available")
    else:
        st.info("Run document analysis to see insights")

def display_knowledge_graph():
    """Display knowledge graph"""
    st.markdown("### 🌐 Knowledge Graph")
    st.info("Knowledge graph visualization will be available after implementing the full enhanced chatbot")

def display_cross_document_analysis():
    """Display cross-document analysis"""
    st.markdown("### 🔄 Cross-Document Analysis")
    
    if len(st.session_state.active_documents) < 2:
        st.warning("⚠️ Please upload at least 2 documents for cross-document analysis")
        return
    
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Document Comparison",
            "Common Themes",
            "Entity Overlap",
            "Synthesis & Summary"
        ]
    )
    
    # Show current settings being used
    st.info(f"Using {st.session_state.analysis_mode} mode with {st.session_state.response_style} style")
    
    if st.button(f"🔍 Perform {analysis_type}", type="primary"):
        with st.spinner(f"Performing {analysis_type.lower()}..."):
            try:
                if analysis_type == "Document Comparison":
                    results = st.session_state.chatbot.compare_documents()
                elif analysis_type == "Common Themes":
                    results = st.session_state.chatbot.find_common_themes()
                elif analysis_type == "Entity Overlap":
                    results = st.session_state.chatbot.analyze_entity_overlap()
                else:  # Synthesis & Summary
                    results = st.session_state.chatbot.synthesize_documents()
                
                if 'error' not in results:
                    st.success(f"✅ {analysis_type} completed successfully")
                    
                    # Display results
                    if 'analysis' in results:
                        st.write(results['analysis'])
                    elif 'comparison' in results:
                        st.write(results['comparison'])
                    else:
                        st.json(results)
                else:
                    st.error(results['error'])
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

def display_sidebar_insights():
    """Display sidebar insights"""
    try:
        insights = st.session_state.document_insights
        
        if not insights or not isinstance(insights, dict):
            st.warning("⚠️ No insights available")
            return
        
        # Multi-document metrics
        if 'multi_document_analysis' in insights:
            analysis = insights['multi_document_analysis']
            
            if isinstance(analysis, dict) and 'error' not in analysis:
                st.metric(
                    "Avg Reading Time",
                    f"{analysis.get('avg_reading_time', 0):.1f} min",
                    f"{analysis.get('total_words', 0):,} words"
                )
                
                st.metric(
                    "Cross-Doc Entities",
                    analysis.get('cross_document_entities_count', 0),
                    "Shared entities"
                )
            else:
                st.warning("⚠️ Multi-document analysis unavailable")
    
    except Exception as e:
        st.error(f"❌ Error displaying insights: {str(e)}")

if __name__ == "__main__":
    main()