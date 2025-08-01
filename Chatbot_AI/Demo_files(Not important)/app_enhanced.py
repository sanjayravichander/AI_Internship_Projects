#!/usr/bin/env python3
"""
Enhanced Elite Streamlit UI for Advanced Multi-Document Q&A Chatbot
Features: Multi-document support, Language translation, Hybrid search, Fixed filter controls
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
from googletrans import Translator
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
    
    /* Language selector */
    .language-selector {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
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
    
    .translated-message {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        margin-right: 20%;
        border: 2px solid rgba(255,255,255,0.2);
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
    if 'translator' not in st.session_state:
        st.session_state.translator = Translator()
    if 'target_language' not in st.session_state:
        st.session_state.target_language = 'en'
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "Comprehensive"
    if 'response_style' not in st.session_state:
        st.session_state.response_style = "Professional"
    if 'search_type' not in st.session_state:
        st.session_state.search_type = "hybrid"
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "multilingual-e5-large"

# Language mapping for translation
LANGUAGE_MAP = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Russian': 'ru',
    'Portuguese': 'pt',
    'Italian': 'it'
}

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
    
    # Enhanced sidebar with multi-document management
    with st.sidebar:
        display_document_management()
        display_enhanced_settings()
        display_language_settings()
        
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
    """Enhanced settings with proper state management"""
    st.markdown("## ⚙️ Advanced Settings")
    
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

def display_language_settings():
    """Language translation settings"""
    st.markdown("## 🌐 Language Settings")
    
    # Target language selection
    target_lang_name = st.selectbox(
        "Translation Language",
        list(LANGUAGE_MAP.keys()),
        index=list(LANGUAGE_MAP.values()).index(st.session_state.target_language),
        help="Select target language for translation"
    )
    
    st.session_state.target_language = LANGUAGE_MAP[target_lang_name]
    
    # Translation toggle
    enable_translation = st.checkbox(
        "Enable Auto-Translation",
        value=st.session_state.target_language != 'en',
        help="Automatically translate responses to selected language"
    )
    
    if enable_translation and st.session_state.target_language == 'en':
        st.session_state.target_language = 'hi'  # Default to Hindi if translation enabled
    elif not enable_translation:
        st.session_state.target_language = 'en'
    
    # Display current language info
    if st.session_state.target_language != 'en':
        st.info(f"🌐 Responses will be translated to {target_lang_name}")

def display_document_insights_panel():
    """Document insights panel in sidebar"""
    st.markdown("## 📊 Document Insights")
    
    if st.button("🔍 Analyze All Documents"):
        with st.spinner("Performing multi-document analysis..."):
            st.session_state.document_insights = st.session_state.chatbot.get_multi_document_insights()
    
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
                <li>Multi-language support (Hindi included)</li>
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
            <h3>🌐 Language Intelligence</h3>
            <ul>
                <li>12+ language support</li>
                <li>Real-time translation</li>
                <li>Hindi language optimization</li>
                <li>Cross-language understanding</li>
                <li>Cultural context awareness</li>
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
        2. **Language Settings**: Set your preferred language before uploading documents
        3. **Search Strategy**: Use 'hybrid' for best results, 'semantic' for conceptual queries
        4. **Embedding Model**: 'multilingual-e5-large' recommended for multi-language support
        5. **Analysis Mode**: Use 'Cross-Document' for comparing multiple documents
        
        **Supported Features:**
        - ✅ PDF, TXT, DOCX files
        - ✅ Multiple documents simultaneously
        - ✅ 12+ languages including Hindi
        - ✅ Hybrid search (semantic + keyword)
        - ✅ Cross-document comparison
        - ✅ Real-time translation
        """)

def display_enhanced_chat_interface():
    """Enhanced chat interface with translation support"""
    
    # Chat history display with translation
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
                # Original AI response
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>Elite AI:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')} • Mode: {message.get('mode', 'AI')} • Search: {message.get('search_type', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Translated response if available
                if 'translated_content' in message and message['translated_content']:
                    lang_name = [k for k, v in LANGUAGE_MAP.items() if v == message.get('target_language', 'en')][0]
                    st.markdown(f"""
                    <div class="chat-message translated-message">
                        <strong>🌐 Translated ({lang_name}):</strong> {message['translated_content']}
                        <br><small>Auto-translated response</small>
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
    
    # Enhanced chat input with language detection
    user_question = st.chat_input("Ask anything about your documents in any language...")
    
    if user_question:
        process_enhanced_question(user_question)

def display_hybrid_search_interface():
    """Enhanced search interface with hybrid capabilities"""
    
    st.markdown("### 🔍 Hybrid Search Engine")
    st.markdown("Combines semantic understanding with keyword matching for optimal results")
    
    col1, col2, col3 = st.columns([4, 2, 2])
    
    with col1:
        search_query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., machine learning algorithms in Hindi: मशीन लर्निंग एल्गोरिदम"
        )
    
    with col2:
        search_strategy = st.selectbox(
            "Search Strategy",
            ["hybrid", "semantic", "keyword", "dense", "sparse"],
            help="Hybrid combines multiple search methods"
        )
    
    with col3:
        max_results = st.number_input("Max Results", min_value=1, max_value=20, value=5)
    
    if search_query and st.button("🚀 Search", type="primary"):
        with st.spinner(f"Performing {search_strategy} search across all documents..."):
            try:
                results = st.session_state.chatbot.perform_hybrid_search(
                    query=search_query,
                    search_type=search_strategy,
                    max_results=max_results
                )
                
                if 'error' not in results:
                    st.success(f"Found {results['total_results']} relevant results using {search_strategy} search")
                    
                    # Display search type indicator
                    search_type_class = f"{search_strategy}-search"
                    st.markdown(f"""
                    <span class="search-type-indicator {search_type_class}">
                        {search_strategy.upper()} SEARCH
                    </span>
                    """, unsafe_allow_html=True)
                    
                    # Display results with enhanced information
                    for i, result in enumerate(results['results']):
                        with st.expander(f"📄 Result #{i+1} - {result.get('document_name', 'Unknown')} (Score: {result['relevance_score']:.3f})"):
                            
                            # Result content
                            st.write("**Content:**")
                            st.write(result['content'])
                            
                            # Metadata
                            if result.get('metadata'):
                                st.write("**Metadata:**")
                                metadata_df = pd.DataFrame([result['metadata']])
                                st.dataframe(metadata_df, use_container_width=True)
                            
                            # Translation option
                            if st.session_state.target_language != 'en':
                                if st.button(f"🌐 Translate to {[k for k, v in LANGUAGE_MAP.items() if v == st.session_state.target_language][0]}", key=f"translate_result_{i}"):
                                    translated = translate_text(result['content'], st.session_state.target_language)
                                    if translated:
                                        st.write("**Translated:**")
                                        st.write(translated)
                else:
                    st.error(results['error'])
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    # Search analytics
    if st.session_state.chatbot:
        st.markdown("### 📊 Search Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Documents", len(st.session_state.active_documents))
        
        with col2:
            total_chunks = sum([doc.get('chunks', 0) for doc in st.session_state.active_documents])
            st.metric("Total Chunks", total_chunks)
        
        with col3:
            st.metric("Search Strategy", st.session_state.search_type.title())
        
        with col4:
            st.metric("Embedding Model", st.session_state.embedding_model.split('/')[-1])

def display_cross_document_analysis():
    """Cross-document analysis interface"""
    
    st.markdown("### 🔄 Cross-Document Analysis")
    st.markdown("Analyze relationships, similarities, and differences across multiple documents")
    
    if len(st.session_state.active_documents) < 2:
        st.warning("⚠️ Please upload at least 2 documents for cross-document analysis")
        return
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Document Comparison",
            "Common Themes",
            "Entity Overlap",
            "Contradiction Detection",
            "Synthesis & Summary",
            "Timeline Analysis"
        ]
    )
    
    if st.button(f"🔍 Perform {analysis_type}", type="primary"):
        with st.spinner(f"Performing {analysis_type.lower()}..."):
            try:
                if analysis_type == "Document Comparison":
                    results = st.session_state.chatbot.compare_documents()
                elif analysis_type == "Common Themes":
                    results = st.session_state.chatbot.find_common_themes()
                elif analysis_type == "Entity Overlap":
                    results = st.session_state.chatbot.analyze_entity_overlap()
                elif analysis_type == "Contradiction Detection":
                    results = st.session_state.chatbot.detect_contradictions()
                elif analysis_type == "Synthesis & Summary":
                    results = st.session_state.chatbot.synthesize_documents()
                else:  # Timeline Analysis
                    results = st.session_state.chatbot.analyze_timeline()
                
                # Display results
                if 'error' not in results:
                    st.success(f"✅ {analysis_type} completed successfully")
                    
                    # Format and display results based on type
                    if analysis_type == "Document Comparison":
                        display_comparison_results(results)
                    elif analysis_type == "Common Themes":
                        display_theme_results(results)
                    elif analysis_type == "Entity Overlap":
                        display_entity_overlap_results(results)
                    else:
                        st.write(results.get('analysis', 'No analysis available'))
                        
                        # Translation option
                        if st.session_state.target_language != 'en':
                            if st.button("🌐 Translate Results"):
                                translated = translate_text(results.get('analysis', ''), st.session_state.target_language)
                                if translated:
                                    st.write("**Translated Results:**")
                                    st.write(translated)
                else:
                    st.error(results['error'])
                    
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

def display_comparison_results(results):
    """Display document comparison results"""
    if 'comparisons' in results:
        for comparison in results['comparisons']:
            with st.expander(f"📊 {comparison['doc1']} vs {comparison['doc2']}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Similarities:**")
                    for sim in comparison.get('similarities', []):
                        st.write(f"• {sim}")
                
                with col2:
                    st.write("**Differences:**")
                    for diff in comparison.get('differences', []):
                        st.write(f"• {diff}")
                
                # Similarity score
                if 'similarity_score' in comparison:
                    st.metric("Similarity Score", f"{comparison['similarity_score']:.2f}")

def display_theme_results(results):
    """Display common themes results"""
    if 'themes' in results:
        st.write("**Common Themes Across Documents:**")
        
        for i, theme in enumerate(results['themes'], 1):
            st.write(f"**{i}. {theme['name']}**")
            st.write(f"   - Frequency: {theme.get('frequency', 'N/A')}")
            st.write(f"   - Documents: {', '.join(theme.get('documents', []))}")
            st.write(f"   - Description: {theme.get('description', 'N/A')}")
            st.write("")

def display_entity_overlap_results(results):
    """Display entity overlap results"""
    if 'entity_overlap' in results:
        # Create visualization
        overlap_data = results['entity_overlap']
        
        # Entity overlap chart
        entities = list(overlap_data.keys())
        counts = [len(overlap_data[entity]['documents']) for entity in entities]
        
        fig = px.bar(
            x=entities[:10],  # Top 10 entities
            y=counts[:10],
            title="Top Entities Across Documents",
            labels={'x': 'Entities', 'y': 'Document Count'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed entity information
        for entity, info in list(overlap_data.items())[:5]:
            with st.expander(f"🏷️ {entity} (appears in {len(info['documents'])} documents)"):
                st.write(f"**Documents:** {', '.join(info['documents'])}")
                st.write(f"**Type:** {info.get('type', 'Unknown')}")
                if 'contexts' in info:
                    st.write("**Contexts:**")
                    for context in info['contexts'][:3]:
                        st.write(f"• {context}")

def translate_text(text: str, target_language: str) -> Optional[str]:
    """Translate text to target language"""
    try:
        if target_language == 'en' or not text:
            return text
        
        translated = st.session_state.translator.translate(text, dest=target_language)
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def process_enhanced_question(question: str):
    """Process user question with enhanced features including translation"""
    
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
            
            # Translate response if needed
            translated_content = None
            if st.session_state.target_language != 'en':
                translated_content = translate_text(
                    response.get('answer', ''), 
                    st.session_state.target_language
                )
            
            # Add AI response
            st.session_state.chat_history.append({
                'content': response.get('answer', 'No response generated'),
                'translated_content': translated_content,
                'target_language': st.session_state.target_language,
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': response.get('mode', 'AI'),
                'search_type': response.get('search_type', st.session_state.search_type),
                'response_time': response_time,
                'documents_used': response.get('documents_used', [])
            })
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            translated_error = None
            
            if st.session_state.target_language != 'en':
                translated_error = translate_text(error_msg, st.session_state.target_language)
            
            st.session_state.chat_history.append({
                'content': error_msg,
                'translated_content': translated_error,
                'target_language': st.session_state.target_language,
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': 'Error'
            })
    
    st.rerun()

# Keep existing functions from original app.py for compatibility
def display_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    
    if not st.session_state.document_insights:
        st.info("📊 Upload documents and run analysis to see the dashboard")
        return
    
    insights = st.session_state.document_insights
    
    # Enhanced metrics for multi-document analysis
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Documents", len(st.session_state.active_documents))
    
    with col2:
        total_pages = sum([doc.get('pages', 0) for doc in st.session_state.active_documents])
        st.metric("Total Pages", total_pages)
    
    with col3:
        if 'multi_document_analysis' in insights:
            total_entities = insights['multi_document_analysis'].get('total_unique_entities', 0)
            st.metric("Unique Entities", total_entities)
    
    with col4:
        st.metric("Search Strategy", st.session_state.search_type.title())
    
    # Multi-document complexity analysis
    if 'multi_document_analysis' in insights:
        st.markdown("### 📈 Multi-Document Analysis")
        
        analysis = insights['multi_document_analysis']
        
        # Document similarity matrix
        if 'similarity_matrix' in analysis:
            st.markdown("#### Document Similarity Matrix")
            similarity_df = pd.DataFrame(analysis['similarity_matrix'])
            fig = px.imshow(
                similarity_df,
                title="Document Similarity Heatmap",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cross-document entity distribution
        if 'cross_document_entities' in analysis:
            st.markdown("#### Cross-Document Entity Distribution")
            entity_data = analysis['cross_document_entities']
            
            entities = list(entity_data.keys())[:10]
            doc_counts = [len(entity_data[entity]) for entity in entities]
            
            fig = px.bar(
                x=entities,
                y=doc_counts,
                title="Top Entities Across Documents",
                labels={'x': 'Entities', 'y': 'Document Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_document_insights():
    """Display comprehensive document insights"""
    
    if not st.session_state.document_insights:
        st.info("📊 Run document analysis to see insights")
        return
    
    insights = st.session_state.document_insights
    
    # Multi-document classification
    if 'multi_document_classification' in insights:
        st.markdown("### 📋 Multi-Document Classification")
        
        classification = insights['multi_document_classification']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Document Types:**")
            for doc_name, doc_type in classification.get('document_types', {}).items():
                st.write(f"• {doc_name}: {doc_type}")
        
        with col2:
            st.write("**Collection Type:**")
            st.info(f"**Type:** {classification.get('collection_type', 'Mixed')}")
            st.info(f"**Coherence:** {classification.get('coherence_score', 'N/A')}")
    
    # Generate multi-document summaries
    st.markdown("### 📝 Multi-Document Summaries")
    
    summary_type = st.selectbox(
        "Summary Type",
        ["executive", "technical", "detailed", "comparative"],
        format_func=lambda x: x.title() + " Summary"
    )
    
    if st.button(f"Generate {summary_type.title()} Summary"):
        with st.spinner(f"Generating {summary_type} summary across all documents..."):
            try:
                summary = st.session_state.chatbot.generate_multi_document_summary(summary_type)
                st.markdown("#### Generated Summary")
                st.write(summary)
                
                # Translation option
                if st.session_state.target_language != 'en':
                    if st.button("🌐 Translate Summary"):
                        translated = translate_text(summary, st.session_state.target_language)
                        if translated:
                            st.markdown("#### Translated Summary")
                            st.write(translated)
                
                # Download button
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{summary_type}_multi_document_summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

def display_knowledge_graph():
    """Display interactive knowledge graph for multi-documents"""
    
    st.markdown("### 🌐 Multi-Document Knowledge Graph")
    
    if st.button("🚀 Generate Cross-Document Knowledge Graph"):
        with st.spinner("Building multi-document knowledge graph..."):
            try:
                graph_data = st.session_state.chatbot.generate_multi_document_knowledge_graph()
                
                if 'error' not in graph_data:
                    # Display graph metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Entities", graph_data.get('total_entities', 0))
                    with col2:
                        st.metric("Cross-Doc Relationships", graph_data.get('cross_doc_relationships', 0))
                    with col3:
                        st.metric("Document Clusters", graph_data.get('document_clusters', 0))
                    with col4:
                        st.metric("Connected Components", graph_data.get('connected_components', 0))
                    
                    # Create enhanced network visualization
                    if graph_data.get('nodes') and graph_data.get('edges'):
                        create_enhanced_network_visualization(graph_data)
                    else:
                        st.info("No significant cross-document relationships found for visualization")
                else:
                    st.error(graph_data['error'])
            except Exception as e:
                st.error(f"Error generating knowledge graph: {str(e)}")

def create_enhanced_network_visualization(graph_data):
    """Create enhanced network visualization for multi-documents"""
    
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes with document information
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges with weights
    for edge in edges:
        if edge['source'] in G.nodes() and edge['target'] in G.nodes():
            G.add_edge(edge['source'], edge['target'], 
                      weight=edge['weight'],
                      documents=edge.get('documents', []))
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Create plotly figure with enhanced features
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge information
        docs = edge[2].get('documents', [])
        edge_info.append(f"Connection: {edge[0]} ↔ {edge[1]}<br>Documents: {', '.join(docs)}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_info = []
    
    # Enhanced color mapping for multi-document entities
    color_map = {
        'PERSON': '#ff7f0e',
        'ORG': '#2ca02c',
        'GPE': '#d62728',
        'DATE': '#9467bd',
        'MONEY': '#8c564b',
        'PRODUCT': '#e377c2',
        'EVENT': '#7f7f7f',
        'LAW': '#bcbd22',
        'LANGUAGE': '#17becf',
        'CROSS_DOC': '#ff1744',  # Special color for cross-document entities
        'UNKNOWN': '#1f77b4'
    }
    
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[0])
        
        # Enhanced node information
        entity_type = node[1].get('type', 'UNKNOWN')
        documents = node[1].get('documents', [])
        frequency = node[1].get('frequency', 1)
        
        # Special handling for cross-document entities
        if len(documents) > 1:
            entity_type = 'CROSS_DOC'
        
        node_color.append(color_map.get(entity_type, '#1f77b4'))
        
        node_info.append(
            f"Entity: {node[0]}<br>"
            f"Type: {entity_type}<br>"
            f"Documents: {', '.join(documents)}<br>"
            f"Frequency: {frequency}"
        )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=[15 if len(G.nodes[node].get('documents', [])) > 1 else 10 for node in G.nodes()],
            color=node_color,
            line=dict(width=2, color='white'),
            opacity=0.8
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Multi-Document Knowledge Graph',
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Interactive Multi-Document Knowledge Graph<br>Red nodes appear in multiple documents",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color='#888', size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend for colors
    st.markdown("#### 🎨 Entity Type Legend")
    legend_cols = st.columns(4)
    
    legend_items = [
        ("PERSON", "👤 Person", '#ff7f0e'),
        ("ORG", "🏢 Organization", '#2ca02c'),
        ("GPE", "🌍 Location", '#d62728'),
        ("CROSS_DOC", "🔗 Cross-Document", '#ff1744')
    ]
    
    for i, (type_key, label, color) in enumerate(legend_items):
        with legend_cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 20px; background-color: {color}; 
                           border-radius: 50%; margin-right: 10px; border: 2px solid white;"></div>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)

def display_sidebar_insights():
    """Display quick insights in sidebar for multi-documents"""
    
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
                    f"{analysis.get('avg_reading_time', 0)} min",
                    f"{analysis.get('total_words', 0):,} words"
                )
                
                st.metric(
                    "Document Similarity",
                    f"{analysis.get('avg_similarity', 0):.2f}",
                    "Average across docs"
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