#!/usr/bin/env python3
"""
Elite Streamlit UI for Advanced Document Q&A Chatbot
This is an enterprise-grade interface showcasing advanced AI/ML capabilities
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
from typing import Dict, List, Any
from agentic_doc_qa_chatbot import AgenticDocumentQAChatbot
from rate_limit_handler import handle_groq_error
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="🚀 Elite Document Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for elite styling
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_insights' not in st.session_state:
    st.session_state.document_insights = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Chat"

def main():
    """Main application function"""
    
    # Elite header
    st.markdown("""
    <div class="elite-header">
        <h1>🚀 Elite Document Intelligence Platform</h1>
        <p>Advanced AI-Powered Document Analysis & Q&A System</p>
        <p><em>Enterprise-Grade Features • Real-time Analytics • Multi-Modal AI</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'txt', 'docx'],
            help="Supports PDF, TXT, and DOCX files"
        )
        
        if uploaded_file:
            if st.button("🚀 Initialize Elite AI", type="primary"):
                initialize_chatbot(uploaded_file)
        
        # Advanced settings
        st.markdown("## ⚙️ Advanced Settings")
        
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Comprehensive", "Fast", "Deep Analysis"],
            help="Choose analysis depth vs speed"
        )
        
        response_style = st.selectbox(
            "Response Style",
            ["Professional", "Technical", "Executive", "Academic"],
            help="Customize response tone and detail level"
        )
        
        # Document insights panel
        if st.session_state.chatbot:
            st.markdown("## 📊 Document Insights")
            
            if st.button("🔍 Analyze Document"):
                with st.spinner("Performing advanced analysis..."):
                    st.session_state.document_insights = st.session_state.chatbot.get_document_insights()
            
            if st.session_state.document_insights:
                display_sidebar_insights()
    
    # Main content area with tabs
    if st.session_state.chatbot:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Intelligent Chat", 
            "📊 Analytics Dashboard", 
            "🔍 Advanced Search", 
            "📈 Document Insights",
            "🌐 Knowledge Graph"
        ])
        
        with tab1:
            display_chat_interface()
        
        with tab2:
            display_analytics_dashboard()
        
        with tab3:
            display_advanced_search()
        
        with tab4:
            display_document_insights()
        
        with tab5:
            display_knowledge_graph()
    
    else:
        display_welcome_screen()

def initialize_chatbot(uploaded_file):
    """Initialize the elite chatbot with uploaded document"""
    try:
        # Save uploaded file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize chatbot with progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing Elite AI System...")
        progress_bar.progress(20)
        
        # Create chatbot instance
        st.session_state.chatbot = AgenticDocumentQAChatbot(
            temp_path, 
            force_recreate=True
        )
        
        progress_bar.progress(60)
        status_text.text("🧠 Loading Advanced AI Models...")
        
        # Initialize advanced features
        time.sleep(1)  # Simulate processing time
        
        progress_bar.progress(80)
        status_text.text("📊 Generating Document Insights...")
        
        # Get initial insights with error handling
        try:
            st.session_state.document_insights = st.session_state.chatbot.get_document_insights()
        except Exception as e:
            st.error(f"❌ Error getting document insights: {str(e)}")
            st.session_state.document_insights = {
                "basic_info": {"error": "Could not load document insights"},
                "complexity_analysis": {"error": "Analysis unavailable"},
                "document_classification": {"error": "Classification unavailable"},
                "key_entities": {"error": "Entity extraction unavailable"},
                "citations_references": {"error": "Citation analysis unavailable"}
            }
        
        progress_bar.progress(100)
        status_text.text("✅ Elite AI System Ready!")
        
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass  # Ignore cleanup errors
        
        st.success("🚀 Elite Document Intelligence Platform is now active!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error initializing system: {str(e)}")

def display_welcome_screen():
    """Display welcome screen with feature highlights"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="elite-card">
            <h3>🧠 Advanced AI Analysis</h3>
            <ul>
                <li>Multi-modal document understanding</li>
                <li>Entity extraction & classification</li>
                <li>Sentiment & complexity analysis</li>
                <li>Topic modeling & clustering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="elite-card">
            <h3>🔍 Intelligent Search</h3>
            <ul>
                <li>Semantic search capabilities</li>
                <li>Citation & reference extraction</li>
                <li>Cross-document comparison</li>
                <li>Smart question suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="elite-card">
            <h3>📊 Enterprise Analytics</h3>
            <ul>
                <li>Real-time insights dashboard</li>
                <li>Knowledge graph visualization</li>
                <li>Document complexity metrics</li>
                <li>Interactive data exploration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Get Started")
    st.info("Upload a document in the sidebar to begin your elite document analysis experience!")

def display_chat_interface():
    """Display the enhanced chat interface"""
    
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
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>Elite AI:</strong> {message['content']}
                    <br><small>{message['timestamp'].strftime('%H:%M:%S')} • Mode: {message.get('mode', 'AI')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Smart suggestions
    st.markdown("#### 🎯 Smart Suggestions")
    
    try:
        suggestions = st.session_state.chatbot.get_suggested_questions()
        if suggestions:
            cols = st.columns(min(3, len(suggestions)))
            for i, suggestion in enumerate(suggestions[:6]):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                        process_question(suggestion)
    except:
        st.info("Smart suggestions will appear after document analysis")
    
    # Chat input
    user_question = st.chat_input("Ask anything about your document...")
    
    if user_question:
        process_question(user_question)

def display_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    
    if not st.session_state.document_insights:
        st.info("📊 Upload a document and run analysis to see the dashboard")
        return
    
    insights = st.session_state.document_insights
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        complexity = insights.get('complexity_analysis', {})
        st.metric(
            "Reading Level",
            complexity.get('complexity_level', 'Unknown'),
            f"Grade {complexity.get('flesch_kincaid_grade', 0)}"
        )
    
    with col2:
        basic_info = insights.get('basic_info', {})
        st.metric(
            "Document Pages",
            basic_info.get('total_pages', 0),
            f"{basic_info.get('total_content_length', 0):,} chars"
        )
    
    with col3:
        entities = insights.get('key_entities', {})
        total_entities = sum(len(v) for v in entities.values() if isinstance(v, list))
        st.metric(
            "Key Entities",
            total_entities,
            "Extracted"
        )
    
    with col4:
        classification = insights.get('document_classification', {})
        st.metric(
            "Document Type",
            classification.get('document_type', 'Unknown')[:15],
            classification.get('confidence', 'Low')
        )
    
    # Complexity analysis chart
    if 'complexity_analysis' in insights and 'error' not in insights['complexity_analysis']:
        st.markdown("### 📈 Document Complexity Analysis")
        
        complexity = insights['complexity_analysis']
        
        # Create complexity radar chart
        categories = ['Reading Ease', 'Vocabulary Richness', 'Sentence Complexity', 'Word Complexity']
        values = [
            complexity.get('flesch_reading_ease', 0) / 100,
            complexity.get('vocabulary_richness', 0),
            min(complexity.get('avg_sentence_length', 0) / 30, 1),  # Normalize to 0-1
            min(complexity.get('avg_word_length', 0) / 10, 1)  # Normalize to 0-1
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Document Complexity'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Document Complexity Profile"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity distribution
    if 'key_entities' in insights and 'error' not in insights['key_entities']:
        st.markdown("### 🏷️ Entity Distribution")
        
        entities = insights['key_entities']
        entity_counts = {k: len(v) for k, v in entities.items() if isinstance(v, list) and v}
        
        if entity_counts:
            fig = px.bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                title="Named Entity Distribution",
                color=list(entity_counts.values()),
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

def display_advanced_search():
    """Display advanced search interface"""
    
    st.markdown("### 🔍 Advanced Semantic Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Enter your search query:", placeholder="e.g., machine learning algorithms")
    
    with col2:
        search_type = st.selectbox("Search Type", ["comprehensive", "precise", "exploratory"])
    
    if search_query and st.button("🚀 Search", type="primary"):
        with st.spinner("Performing semantic search..."):
            try:
                results = st.session_state.chatbot.perform_semantic_search(search_query, search_type)
                
                if 'error' not in results:
                    st.success(f"Found {results['total_results']} relevant results")
                    
                    for result in results['results'][:5]:  # Show top 5 results
                        with st.expander(f"Result #{result['rank']} (Relevance: {result['relevance_score']:.2f})"):
                            st.write(result['content'])
                            if result['metadata']:
                                st.json(result['metadata'])
                else:
                    st.error(results['error'])
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    # Citation and reference analysis
    st.markdown("### 📚 Citations & References")
    
    if st.button("🔍 Extract Citations"):
        with st.spinner("Analyzing citations and references..."):
            try:
                citations = st.session_state.chatbot.find_citations_and_references()
                
                if 'error' not in citations:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Citations", len(citations.get('citations', [])))
                        if citations.get('citations'):
                            with st.expander("View Citations"):
                                for citation in citations['citations'][:10]:
                                    st.write(f"• {citation}")
                    
                    with col2:
                        st.metric("URLs", len(citations.get('urls', [])))
                        if citations.get('urls'):
                            with st.expander("View URLs"):
                                for url in citations['urls'][:10]:
                                    st.write(f"• {url}")
                    
                    with col3:
                        st.metric("DOIs", len(citations.get('dois', [])))
                        if citations.get('dois'):
                            with st.expander("View DOIs"):
                                for doi in citations['dois'][:10]:
                                    st.write(f"• {doi}")
                else:
                    st.error(citations['error'])
            except Exception as e:
                st.error(f"Citation analysis error: {str(e)}")

def display_document_insights():
    """Display comprehensive document insights"""
    
    if not st.session_state.document_insights:
        st.info("📊 Run document analysis to see insights")
        return
    
    insights = st.session_state.document_insights
    
    # Document classification
    if 'document_classification' in insights:
        classification = insights['document_classification']
        if 'error' not in classification:
            st.markdown("### 📋 Document Classification")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Type:** {classification.get('document_type', 'Unknown')}")
                st.info(f"**Confidence:** {classification.get('confidence', 'Unknown')}")
            
            with col2:
                if classification.get('reasoning'):
                    st.write("**Reasoning:**")
                    st.write(classification['reasoning'])
    
    # Generate different summary types
    st.markdown("### 📝 AI-Generated Summaries")
    
    summary_type = st.selectbox(
        "Summary Type",
        ["executive", "technical", "detailed"],
        format_func=lambda x: x.title() + " Summary"
    )
    
    if st.button(f"Generate {summary_type.title()} Summary"):
        with st.spinner(f"Generating {summary_type} summary..."):
            try:
                summary = st.session_state.chatbot.generate_executive_summary(summary_type)
                st.markdown("#### Generated Summary")
                st.write(summary)
                
                # Add download button
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name=f"{summary_type}_summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

def display_knowledge_graph():
    """Display interactive knowledge graph"""
    
    st.markdown("### 🌐 Knowledge Graph Visualization")
    
    if st.button("🚀 Generate Knowledge Graph"):
        with st.spinner("Building knowledge graph..."):
            try:
                graph_data = st.session_state.chatbot.generate_knowledge_graph()
                
                if 'error' not in graph_data:
                    # Display graph metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entities", graph_data.get('connected_entities', 0))
                    with col2:
                        st.metric("Relationships", graph_data.get('relationships', 0))
                    with col3:
                        st.metric("Total Extracted", graph_data.get('total_entities', 0))
                    
                    # Create network visualization
                    if graph_data.get('nodes') and graph_data.get('edges'):
                        create_network_visualization(graph_data)
                    else:
                        st.info("No significant relationships found for visualization")
                else:
                    st.error(graph_data['error'])
            except Exception as e:
                st.error(f"Error generating knowledge graph: {str(e)}")

def create_network_visualization(graph_data):
    """Create interactive network visualization"""
    
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges
    for edge in edges:
        if edge['source'] in G.nodes() and edge['target'] in G.nodes():
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    
    # Calculate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        # Color by entity type
        node_info = G.nodes[node]
        entity_type = node_info.get('type', 'UNKNOWN')
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
            'UNKNOWN': '#1f77b4'
        }
        node_color.append(color_map.get(entity_type, '#1f77b4'))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=10,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Document Knowledge Graph',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="Interactive Knowledge Graph - Hover over nodes for details",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color='#888', size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    st.plotly_chart(fig, use_container_width=True)

def display_sidebar_insights():
    """Display quick insights in sidebar"""
    
    try:
        insights = st.session_state.document_insights
        
        if not insights or not isinstance(insights, dict):
            st.warning("⚠️ No insights available")
            return
        
        if 'complexity_analysis' in insights:
            complexity = insights['complexity_analysis']
            if isinstance(complexity, dict) and 'error' not in complexity:
                st.metric(
                    "Reading Time",
                    f"{complexity.get('estimated_reading_time', 0)} min",
                    f"{complexity.get('total_words', 0):,} words"
                )
            else:
                st.warning("⚠️ Complexity analysis unavailable")
    
    except Exception as e:
        st.error(f"❌ Error displaying insights: {str(e)}")

def process_question(question: str):
    """Process user question with enhanced features"""
    
    # Add user message
    st.session_state.chat_history.append({
        'content': question,
        'is_user': True,
        'timestamp': datetime.now()
    })
    
    # Get AI response
    with st.spinner("🧠 Elite AI is analyzing..."):
        try:
            start_time = time.time()
            response = st.session_state.chatbot.ask_question(question)
            response_time = time.time() - start_time
            
            # Add AI response
            st.session_state.chat_history.append({
                'content': response.get('answer', 'No response generated'),
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': response.get('mode', 'AI'),
                'response_time': response_time
            })
            
        except Exception as e:
            st.session_state.chat_history.append({
                'content': f"Error: {str(e)}",
                'is_user': False,
                'timestamp': datetime.now(),
                'mode': 'Error'
            })
    
    st.rerun()

if __name__ == "__main__":
    main()