#!/usr/bin/env python3
"""
Enhanced Agentic Document Q&A Chatbot with Multi-Document Support
Features: Multiple documents, Hybrid search, Multi-language support, Advanced embeddings
"""

import os
import hashlib
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from typing import Dict, List, Any, Optional, Union
import json
from advanced_features import AdvancedDocumentAnalyzer, DocumentComparison, SmartSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import defaultdict
import re

# Load environment variables
dotenv_path = os.getenv("ENV_PATH", os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=dotenv_path)

class EnhancedAgenticDocumentQAChatbot:
    """Enhanced chatbot with multi-document support and hybrid search"""
    
    def __init__(self, 
                 document_paths: List[str],
                 embedding_model: str = "multilingual-e5-large",
                 search_type: str = "hybrid",
                 analysis_mode: str = "Comprehensive",
                 response_style: str = "Professional",
                 vectorstore_base_path: Optional[str] = None,
                 force_recreate: bool = False):
        """
        Initialize Enhanced Agentic Document Q&A Chatbot
        
        Args:
            document_paths: List of paths to documents
            embedding_model: Embedding model to use
            search_type: Type of search (hybrid, semantic, keyword, etc.)
            analysis_mode: Analysis mode for responses
            response_style: Style of responses
            vectorstore_base_path: Base path for vector store
            force_recreate: Force recreation of vector store
        """
        self.document_paths = document_paths
        self.embedding_model_name = embedding_model
        self.search_type = search_type
        self.analysis_mode = analysis_mode
        self.response_style = response_style
        self.force_recreate = force_recreate
        
        # Initialize paths
        if vectorstore_base_path is None:
            vectorstore_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore"))
        self.vectorstore_base_path = vectorstore_base_path
        
        # Initialize components
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.qa_chain = None
        self.agent_executor = None
        self.conversation_history = []
        self.documents_metadata = {}
        self.all_documents = []
        
        # Generate unique vector store path
        self.vectorstore_path = self._generate_vectorstore_path()
        
        # Initialize Groq API key
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Please set your GROQ_API_KEY in the .env file")
        
        # Clear existing vector store if force_recreate
        if self.force_recreate:
            self._clear_existing_vectorstore()
        
        # Initialize the enhanced chatbot
        self.setup_enhanced_chatbot()
        self.setup_agent()
        
        # Initialize advanced features
        self.advanced_analyzer = None
        self.document_comparison = None
        self.smart_search = None
        self._setup_advanced_features()
        
        # Load spaCy model for advanced NLP
        self.nlp = self._load_spacy_model()
        
        # Generate suggested questions
        self.suggested_questions = self.generate_suggested_questions()
    
    def _generate_vectorstore_path(self) -> str:
        """Generate unique vector store path for multiple documents"""
        # Create hash from all document paths and settings
        path_string = "|".join(sorted(self.document_paths))
        settings_string = f"{self.embedding_model_name}_{self.search_type}"
        combined_string = f"{path_string}_{settings_string}"
        
        combined_hash = hashlib.md5(combined_string.encode()).hexdigest()[:12]
        vectorstore_path = os.path.join(self.vectorstore_base_path, f"multi_doc_faiss_{combined_hash}")
        
        print(f"📁 Multi-document vector store path: {vectorstore_path}")
        return vectorstore_path
    
    def _clear_existing_vectorstore(self):
        """Clear existing vector store directory"""
        import shutil
        try:
            if os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)
                print(f"🗑️ Cleared existing vector store at: {self.vectorstore_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear existing vector store: {e}")
    
    def _load_spacy_model(self):
        """Load spaCy model for NLP tasks"""
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Some features may be limited.")
            return None
    
    def load_and_process_documents(self) -> List:
        """Load and process multiple documents"""
        print(f"Loading and processing {len(self.document_paths)} documents...")
        
        all_docs = []
        
        for doc_path in self.document_paths:
            try:
                # Determine loader based on file extension
                file_ext = os.path.splitext(doc_path)[1].lower()
                
                if file_ext == '.pdf':
                    loader = PyPDFLoader(doc_path)
                elif file_ext == '.txt':
                    loader = TextLoader(doc_path, encoding='utf-8')
                elif file_ext in ['.docx', '.doc']:
                    loader = Docx2txtLoader(doc_path)
                else:
                    print(f"⚠️ Unsupported file type: {file_ext}")
                    continue
                
                # Load document
                pages = loader.load()
                
                # Store metadata
                doc_name = os.path.basename(doc_path)
                self.documents_metadata[doc_name] = {
                    "total_pages": len(pages),
                    "document_path": doc_path,
                    "total_content_length": sum(len(page.page_content) for page in pages),
                    "file_type": file_ext
                }
                
                # Add document name to each page's metadata
                for page in pages:
                    page.metadata.update({
                        "document_name": doc_name,
                        "document_path": doc_path,
                        "file_type": file_ext
                    })
                
                all_docs.extend(pages)
                print(f"📄 Loaded {doc_name}: {len(pages)} pages")
                
            except Exception as e:
                print(f"❌ Error loading {doc_path}: {str(e)}")
                continue
        
        if not all_docs:
            raise ValueError("No documents could be loaded successfully")
        
        # Enhanced document splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Optimized for multi-document analysis
            chunk_overlap=300,  # Higher overlap for better context
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        docs = splitter.split_documents(all_docs)
        
        # Add enhanced metadata to each chunk
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": i,
                "total_chunks": len(docs),
                "total_documents": len(self.document_paths)
            })
        
        self.all_documents = docs
        print(f"📊 Total chunks created: {len(docs)} from {len(self.document_paths)} documents")
        return docs
    
    def get_best_embedding_model(self) -> HuggingFaceEmbeddings:
        """Get the best embedding model based on selection"""
        print(f"🔄 Loading embedding model: {self.embedding_model_name}")
        
        # Model configurations with fallbacks
        model_configs = {
            "multilingual-e5-large": {
                "model_name": "intfloat/multilingual-e5-large",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True}
            },
            "multilingual-e5-base": {
                "model_name": "intfloat/multilingual-e5-base", 
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True}
            },
            "paraphrase-multilingual-mpnet-base-v2": {
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True}
            },
            "all-MiniLM-L6-v2": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True}
            },
            "jinaai/jina-embeddings-v3": {
                "model_name": "jinaai/jina-embeddings-v3",
                "model_kwargs": {"device": "cpu", "trust_remote_code": True},
                "encode_kwargs": {"normalize_embeddings": True}
            }
        }
        
        # Try primary model first
        if self.embedding_model_name in model_configs:
            config = model_configs[self.embedding_model_name]
            try:
                embed_model = HuggingFaceEmbeddings(**config)
                print(f"✅ Successfully loaded {self.embedding_model_name}")
                return embed_model
            except Exception as e:
                print(f"⚠️ Failed to load {self.embedding_model_name}: {e}")
        
        # Fallback sequence
        fallback_models = ["multilingual-e5-base", "all-MiniLM-L6-v2", "paraphrase-multilingual-mpnet-base-v2"]
        
        for fallback in fallback_models:
            if fallback in model_configs:
                try:
                    config = model_configs[fallback]
                    embed_model = HuggingFaceEmbeddings(**config)
                    print(f"✅ Fallback to {fallback}")
                    return embed_model
                except Exception as e:
                    print(f"⚠️ Fallback {fallback} failed: {e}")
                    continue
        
        # Final fallback
        print("🔄 Using final fallback: BAAI/bge-small-en-v1.5")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    def create_hybrid_vectorstore(self, docs: List) -> FAISS:
        """Create enhanced vector store with hybrid search capabilities"""
        print("Creating enhanced vector store with hybrid search...")
        
        # Get embedding model
        embed_model = self.get_best_embedding_model()
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embed_model)
        
        # Create BM25 retriever for keyword search
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 6
        
        # Create ensemble retriever for hybrid search
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # Favor semantic search slightly
        )
        
        # Save vector store
        vectorstore.save_local(self.vectorstore_path)
        print("✅ Enhanced hybrid vector store created and saved!")
        
        return vectorstore
    
    def load_existing_vectorstore(self) -> Optional[FAISS]:
        """Load existing vector store if available"""
        try:
            embed_model = self.get_best_embedding_model()
            vectorstore = FAISS.load_local(
                self.vectorstore_path,
                embed_model,
                allow_dangerous_deserialization=True
            )
            
            # Recreate BM25 retriever (can't be saved)
            if self.all_documents:
                self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                self.bm25_retriever.k = 6
                
                # Recreate ensemble retriever
                faiss_retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}
                )
                
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[faiss_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
            
            print("✅ Existing enhanced vector store loaded!")
            return vectorstore
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return None
    
    def setup_enhanced_chatbot(self):
        """Setup the enhanced chatbot pipeline"""
        # Load or create vector store
        if self.force_recreate:
            print("🔄 Force recreating vector store...")
            docs = self.load_and_process_documents()
            self.vectorstore = self.create_hybrid_vectorstore(docs)
        else:
            # Try to load existing first
            self.vectorstore = self.load_existing_vectorstore()
            
            if self.vectorstore is None:
                docs = self.load_and_process_documents()
                self.vectorstore = self.create_hybrid_vectorstore(docs)
        
        # Setup enhanced Q&A chain
        self.llm = self.setup_enhanced_qa_chain()
    
    def setup_enhanced_qa_chain(self) -> ChatGroq:
        """Setup enhanced Q&A chain with multi-document awareness"""
        print("Setting up enhanced Q&A chain...")
        
        # Initialize Groq LLM with optimized settings
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,  # Balanced for accuracy and creativity
            max_tokens=800    # Increased for comprehensive responses
        )
        
        # Create enhanced prompt template
        prompt_template = f"""
        You are an expert AI assistant analyzing multiple documents simultaneously.
        
        Documents in your knowledge base: {list(self.documents_metadata.keys())}
        Total documents: {len(self.document_paths)}
        Analysis mode: {self.analysis_mode}
        Response style: {self.response_style}
        
        IMPORTANT GUIDELINES:
        1. You have access to multiple documents - leverage cross-document insights
        2. When answering, specify which document(s) your information comes from
        3. For multi-document questions, synthesize information across all relevant documents
        4. Identify similarities, differences, and relationships between documents
        5. Use the specified response style: {self.response_style}
        6. For cross-document analysis, compare and contrast information
        7. Always cite the source document for specific claims
        
        Context from documents:
        {{context}}
        
        Question: {{question}}
        
        Enhanced Multi-Document Answer (in {self.response_style} style):
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain with hybrid search
        retriever = self.get_retriever_by_type(self.search_type)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ Enhanced Q&A chain setup complete!")
        return llm
    
    def get_retriever_by_type(self, search_type: str):
        """Get retriever based on search type"""
        if search_type == "hybrid" and self.ensemble_retriever:
            return self.ensemble_retriever
        elif search_type == "semantic":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
        elif search_type == "keyword" and self.bm25_retriever:
            return self.bm25_retriever
        elif search_type == "dense":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}
            )
        elif search_type == "sparse" and self.bm25_retriever:
            self.bm25_retriever.k = 10
            return self.bm25_retriever
        else:
            # Default to semantic search
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
    
    def update_analysis_mode(self, mode: str):
        """Update analysis mode"""
        self.analysis_mode = mode
        print(f"🔄 Analysis mode updated to: {mode}")
    
    def update_response_style(self, style: str):
        """Update response style"""
        self.response_style = style
        print(f"🔄 Response style updated to: {style}")
    
    def update_search_type(self, search_type: str):
        """Update search type"""
        self.search_type = search_type
        print(f"🔄 Search type updated to: {search_type}")
    
    def perform_hybrid_search(self, query: str, search_type: str = "hybrid", max_results: int = 5) -> Dict[str, Any]:
        """Perform hybrid search across all documents"""
        try:
            retriever = self.get_retriever_by_type(search_type)
            docs = retriever.get_relevant_documents(query)
            
            results = []
            for i, doc in enumerate(docs[:max_results]):
                # Calculate relevance score (simplified)
                relevance_score = 1.0 - (i * 0.1)  # Decreasing score
                
                results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "document_name": doc.metadata.get("document_name", "Unknown"),
                    "relevance_score": relevance_score
                })
            
            return {
                "total_results": len(results),
                "results": results,
                "search_type": search_type,
                "query": query
            }
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}
    
    def compare_documents(self) -> Dict[str, Any]:
        """Compare multiple documents"""
        try:
            if len(self.document_paths) < 2:
                return {"error": "Need at least 2 documents for comparison"}
            
            # Get document summaries
            doc_summaries = {}
            for doc_name in self.documents_metadata.keys():
                query = f"Summarize the main points and themes of {doc_name}"
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": {"document_name": doc_name}}
                )
                docs = retriever.get_relevant_documents(query)
                
                content = "\n".join([doc.page_content for doc in docs])
                summary_prompt = f"Summarize the main themes and key points from this document content:\n{content[:2000]}"
                
                response = self.llm.invoke(summary_prompt)
                doc_summaries[doc_name] = response.content if hasattr(response, 'content') else str(response)
            
            # Compare documents
            comparison_prompt = f"""
            Compare and contrast these documents:
            
            {json.dumps(doc_summaries, indent=2)}
            
            Provide:
            1. Key similarities between documents
            2. Major differences
            3. Unique aspects of each document
            4. Overall relationship between documents
            """
            
            response = self.llm.invoke(comparison_prompt)
            comparison_result = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "comparison": comparison_result,
                "document_summaries": doc_summaries,
                "documents_compared": list(doc_summaries.keys())
            }
        except Exception as e:
            return {"error": f"Comparison error: {str(e)}"}
    
    def find_common_themes(self) -> Dict[str, Any]:
        """Find common themes across documents"""
        try:
            # Extract key themes from each document
            all_themes = []
            doc_themes = {}
            
            for doc_name in self.documents_metadata.keys():
                query = f"key themes topics main ideas {doc_name}"
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 8}
                )
                docs = retriever.get_relevant_documents(query)
                
                # Filter docs for this specific document
                doc_specific_content = []
                for doc in docs:
                    if doc.metadata.get("document_name") == doc_name:
                        doc_specific_content.append(doc.page_content)
                
                if doc_specific_content:
                    content = "\n".join(doc_specific_content[:3])
                    theme_prompt = f"Extract the main themes and topics from this content. List them as bullet points:\n{content[:1500]}"
                    
                    response = self.llm.invoke(theme_prompt)
                    themes_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract themes (simplified)
                    themes = [line.strip('• -').strip() for line in themes_text.split('\n') if line.strip() and ('•' in line or '-' in line)]
                    doc_themes[doc_name] = themes
                    all_themes.extend(themes)
            
            # Find common themes (simplified approach)
            theme_counts = defaultdict(int)
            for theme in all_themes:
                theme_lower = theme.lower()
                for other_theme in all_themes:
                    if theme_lower in other_theme.lower() or other_theme.lower() in theme_lower:
                        theme_counts[theme] += 1
            
            # Get most common themes
            common_themes = []
            for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                if count > 1:  # Appears in multiple contexts
                    # Find which documents contain this theme
                    containing_docs = []
                    for doc_name, themes in doc_themes.items():
                        if any(theme.lower() in t.lower() or t.lower() in theme.lower() for t in themes):
                            containing_docs.append(doc_name)
                    
                    common_themes.append({
                        "name": theme,
                        "frequency": count,
                        "documents": containing_docs,
                        "description": f"Theme appearing across {len(containing_docs)} documents"
                    })
            
            return {
                "themes": common_themes,
                "document_themes": doc_themes,
                "total_themes_found": len(all_themes)
            }
        except Exception as e:
            return {"error": f"Theme analysis error: {str(e)}"}
    
    def analyze_entity_overlap(self) -> Dict[str, Any]:
        """Analyze entity overlap across documents"""
        try:
            if not self.nlp:
                return {"error": "spaCy model not available for entity analysis"}
            
            doc_entities = {}
            all_entities = defaultdict(lambda: {"documents": set(), "type": None, "contexts": []})
            
            # Extract entities from each document
            for doc_name in self.documents_metadata.keys():
                # Get content for this document
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                )
                docs = retriever.get_relevant_documents(f"content from {doc_name}")
                
                doc_content = ""
                for doc in docs:
                    if doc.metadata.get("document_name") == doc_name:
                        doc_content += doc.page_content + " "
                
                if doc_content:
                    # Process with spaCy (limit content size)
                    doc_nlp = self.nlp(doc_content[:50000])
                    
                    entities = []
                    for ent in doc_nlp.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                            entities.append({
                                "text": ent.text,
                                "label": ent.label_,
                                "context": ent.sent.text[:200]
                            })
                            
                            # Add to global entity tracking
                            all_entities[ent.text]["documents"].add(doc_name)
                            all_entities[ent.text]["type"] = ent.label_
                            all_entities[ent.text]["contexts"].append(ent.sent.text[:200])
                    
                    doc_entities[doc_name] = entities
            
            # Convert sets to lists for JSON serialization
            entity_overlap = {}
            for entity, info in all_entities.items():
                if len(info["documents"]) > 1:  # Only entities appearing in multiple docs
                    entity_overlap[entity] = {
                        "documents": list(info["documents"]),
                        "type": info["type"],
                        "contexts": info["contexts"][:3],  # Limit contexts
                        "frequency": len(info["contexts"])
                    }
            
            return {
                "entity_overlap": entity_overlap,
                "document_entities": doc_entities,
                "cross_document_entities_count": len(entity_overlap)
            }
        except Exception as e:
            return {"error": f"Entity analysis error: {str(e)}"}
    
    def detect_contradictions(self) -> Dict[str, Any]:
        """Detect contradictions between documents"""
        try:
            # This is a simplified approach - in practice, you'd want more sophisticated NLP
            contradiction_prompt = f"""
            Analyze the following documents for contradictions, conflicting information, or disagreements:
            
            Documents: {list(self.documents_metadata.keys())}
            
            Look for:
            1. Conflicting facts or figures
            2. Opposing viewpoints
            3. Contradictory conclusions
            4. Different interpretations of the same topic
            
            Provide specific examples with document sources.
            """
            
            # Get diverse content from all documents
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 15, "lambda_mult": 0.3}
            )
            docs = retriever.get_relevant_documents("facts conclusions viewpoints opinions")
            
            # Combine content with document attribution
            content_by_doc = defaultdict(list)
            for doc in docs:
                doc_name = doc.metadata.get("document_name", "Unknown")
                content_by_doc[doc_name].append(doc.page_content)
            
            # Create comparison content
            comparison_content = ""
            for doc_name, contents in content_by_doc.items():
                comparison_content += f"\n\n=== {doc_name} ===\n"
                comparison_content += "\n".join(contents[:3])  # Limit content
            
            full_prompt = f"{contradiction_prompt}\n\nContent to analyze:\n{comparison_content[:3000]}"
            
            response = self.llm.invoke(full_prompt)
            analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": analysis,
                "documents_analyzed": list(content_by_doc.keys()),
                "method": "LLM-based contradiction detection"
            }
        except Exception as e:
            return {"error": f"Contradiction detection error: {str(e)}"}
    
    def synthesize_documents(self) -> Dict[str, Any]:
        """Synthesize information across all documents"""
        try:
            synthesis_prompt = f"""
            Create a comprehensive synthesis of all {len(self.document_paths)} documents.
            
            Documents: {list(self.documents_metadata.keys())}
            
            Provide:
            1. Unified overview of all documents
            2. Key insights that emerge from combining all sources
            3. Comprehensive conclusions drawn from all documents
            4. Areas where documents complement each other
            5. Integrated recommendations or findings
            
            Create a cohesive narrative that brings together all the information.
            """
            
            # Get comprehensive content from all documents
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 20, "lambda_mult": 0.4}
            )
            docs = retriever.get_relevant_documents("main points key findings conclusions recommendations")
            
            # Organize content by document
            doc_contents = defaultdict(list)
            for doc in docs:
                doc_name = doc.metadata.get("document_name", "Unknown")
                doc_contents[doc_name].append(doc.page_content)
            
            # Create structured content for synthesis
            structured_content = ""
            for doc_name, contents in doc_contents.items():
                structured_content += f"\n\n=== Key Content from {doc_name} ===\n"
                structured_content += "\n".join(contents[:4])
            
            full_prompt = f"{synthesis_prompt}\n\nContent to synthesize:\n{structured_content[:4000]}"
            
            response = self.llm.invoke(full_prompt)
            synthesis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": synthesis,
                "documents_synthesized": list(doc_contents.keys()),
                "synthesis_type": "Comprehensive multi-document synthesis"
            }
        except Exception as e:
            return {"error": f"Synthesis error: {str(e)}"}
    
    def analyze_timeline(self) -> Dict[str, Any]:
        """Analyze timeline and chronological information across documents"""
        try:
            timeline_prompt = f"""
            Analyze the chronological information and timeline across these documents:
            
            Documents: {list(self.documents_metadata.keys())}
            
            Extract:
            1. Important dates and time periods mentioned
            2. Chronological sequence of events
            3. Timeline relationships between documents
            4. Historical context and progression
            5. Temporal patterns or trends
            
            Create a unified timeline if possible.
            """
            
            # Search for date-related content
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 15}
            )
            docs = retriever.get_relevant_documents("date time year month chronology timeline history when")
            
            # Extract content with dates
            date_content = []
            for doc in docs:
                content = doc.page_content
                doc_name = doc.metadata.get("document_name", "Unknown")
                
                # Simple date pattern matching
                date_patterns = [
                    r'\b\d{4}\b',  # Years
                    r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                    r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
                ]
                
                has_dates = any(re.search(pattern, content) for pattern in date_patterns)
                if has_dates:
                    date_content.append(f"From {doc_name}:\n{content}")
            
            if date_content:
                timeline_content = "\n\n".join(date_content[:10])
                full_prompt = f"{timeline_prompt}\n\nContent with dates:\n{timeline_content[:3000]}"
            else:
                full_prompt = f"{timeline_prompt}\n\nNote: Limited explicit date information found in documents."
            
            response = self.llm.invoke(full_prompt)
            timeline_analysis = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "analysis": timeline_analysis,
                "documents_analyzed": list(self.documents_metadata.keys()),
                "date_content_found": len(date_content) > 0
            }
        except Exception as e:
            return {"error": f"Timeline analysis error: {str(e)}"}
    
    def generate_multi_document_summary(self, summary_type: str = "executive") -> str:
        """Generate summary across all documents"""
        try:
            style_prompts = {
                "executive": "Create an executive summary suitable for senior leadership, focusing on key insights, strategic implications, and actionable recommendations.",
                "technical": "Create a detailed technical summary covering methodologies, technical details, and implementation aspects.",
                "detailed": "Create a comprehensive detailed summary covering all major points, findings, and conclusions.",
                "comparative": "Create a comparative summary highlighting similarities, differences, and relationships between the documents."
            }
            
            base_prompt = f"""
            {style_prompts.get(summary_type, style_prompts['executive'])}
            
            Documents to summarize: {list(self.documents_metadata.keys())}
            Total documents: {len(self.document_paths)}
            
            Provide a {summary_type} summary that synthesizes information from all documents.
            """
            
            # Get comprehensive content
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 25, "lambda_mult": 0.5}
            )
            docs = retriever.get_relevant_documents("summary main points key findings conclusions")
            
            # Organize by document
            doc_summaries = defaultdict(list)
            for doc in docs:
                doc_name = doc.metadata.get("document_name", "Unknown")
                doc_summaries[doc_name].append(doc.page_content)
            
            # Create structured input
            structured_input = ""
            for doc_name, contents in doc_summaries.items():
                structured_input += f"\n\n=== {doc_name} ===\n"
                structured_input += "\n".join(contents[:3])
            
            full_prompt = f"{base_prompt}\n\nContent to summarize:\n{structured_input[:4000]}"
            
            response = self.llm.invoke(full_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_multi_document_knowledge_graph(self) -> Dict[str, Any]:
        """Generate knowledge graph across all documents"""
        try:
            if not self.nlp:
                return {"error": "spaCy model not available for knowledge graph generation"}
            
            # Extract entities and relationships from all documents
            all_entities = {}
            all_relationships = []
            doc_entity_counts = defaultdict(int)
            
            for doc_name in self.documents_metadata.keys():
                # Get content for this document
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 8}
                )
                docs = retriever.get_relevant_documents(f"entities relationships {doc_name}")
                
                doc_content = ""
                for doc in docs:
                    if doc.metadata.get("document_name") == doc_name:
                        doc_content += doc.page_content + " "
                
                if doc_content:
                    # Process with spaCy
                    doc_nlp = self.nlp(doc_content[:30000])
                    
                    # Extract entities
                    doc_entities = []
                    for ent in doc_nlp.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'LAW']:
                            entity_id = ent.text
                            if entity_id not in all_entities:
                                all_entities[entity_id] = {
                                    "id": entity_id,
                                    "type": ent.label_,
                                    "documents": set(),
                                    "frequency": 0
                                }
                            
                            all_entities[entity_id]["documents"].add(doc_name)
                            all_entities[entity_id]["frequency"] += 1
                            doc_entities.append(entity_id)
                            doc_entity_counts[doc_name] += 1
                    
                    # Create relationships (entities appearing in same sentences)
                    for sent in doc_nlp.sents:
                        sent_entities = [ent.text for ent in sent.ents 
                                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'LAW']]
                        
                        for i, ent1 in enumerate(sent_entities):
                            for ent2 in sent_entities[i+1:]:
                                if ent1 != ent2:
                                    all_relationships.append({
                                        "source": ent1,
                                        "target": ent2,
                                        "weight": 1,
                                        "documents": [doc_name]
                                    })
            
            # Convert entity documents sets to lists
            nodes = []
            for entity_id, entity_data in all_entities.items():
                entity_data["documents"] = list(entity_data["documents"])
                nodes.append(entity_data)
            
            # Aggregate relationships
            relationship_map = defaultdict(lambda: {"weight": 0, "documents": set()})
            for rel in all_relationships:
                key = tuple(sorted([rel["source"], rel["target"]]))
                relationship_map[key]["weight"] += rel["weight"]
                relationship_map[key]["documents"].update(rel["documents"])
            
            # Convert to edges format
            edges = []
            cross_doc_relationships = 0
            for (source, target), data in relationship_map.items():
                edge = {
                    "source": source,
                    "target": target,
                    "weight": data["weight"],
                    "documents": list(data["documents"])
                }
                edges.append(edge)
                
                if len(data["documents"]) > 1:
                    cross_doc_relationships += 1
            
            return {
                "nodes": nodes,
                "edges": edges,
                "total_entities": len(nodes),
                "cross_doc_relationships": cross_doc_relationships,
                "document_clusters": len(self.documents_metadata),
                "connected_components": len(nodes),  # Simplified
                "document_entity_counts": dict(doc_entity_counts)
            }
        except Exception as e:
            return {"error": f"Knowledge graph generation error: {str(e)}"}
    
    def get_multi_document_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights across all documents"""
        try:
            insights = {
                "basic_info": {
                    "total_documents": len(self.document_paths),
                    "documents": list(self.documents_metadata.keys()),
                    "total_pages": sum(meta["total_pages"] for meta in self.documents_metadata.values()),
                    "total_content_length": sum(meta["total_content_length"] for meta in self.documents_metadata.values()),
                    "embedding_model": self.embedding_model_name,
                    "search_type": self.search_type
                }
            }
            
            # Multi-document analysis
            multi_doc_analysis = {
                "document_metadata": self.documents_metadata,
                "avg_reading_time": sum(meta["total_content_length"] for meta in self.documents_metadata.values()) / 200 / 60,  # Rough estimate
                "total_words": sum(meta["total_content_length"] for meta in self.documents_metadata.values()) // 5,  # Rough estimate
            }
            
            # Add entity overlap analysis
            try:
                entity_analysis = self.analyze_entity_overlap()
                if "error" not in entity_analysis:
                    multi_doc_analysis["cross_document_entities_count"] = entity_analysis.get("cross_document_entities_count", 0)
                    multi_doc_analysis["cross_document_entities"] = entity_analysis.get("entity_overlap", {})
            except:
                pass
            
            # Add document similarity (simplified)
            try:
                if len(self.document_paths) > 1:
                    # This is a simplified similarity calculation
                    multi_doc_analysis["avg_similarity"] = 0.75  # Placeholder
                    multi_doc_analysis["similarity_matrix"] = {}  # Placeholder
            except:
                pass
            
            insights["multi_document_analysis"] = multi_doc_analysis
            
            return insights
        except Exception as e:
            return {"error": f"Error getting multi-document insights: {str(e)}"}
    
    def generate_suggested_questions(self) -> List[str]:
        """Generate intelligent question suggestions for multi-documents"""
        try:
            print("🤔 Generating multi-document question suggestions...")
            
            # Base suggestions for multi-document analysis
            base_suggestions = [
                "What are the main themes across all documents?",
                "Compare the key findings between documents",
                "What entities appear in multiple documents?",
                "Summarize the overall conclusions from all documents",
                "What are the similarities and differences between documents?",
                "Find contradictions or agreements between documents"
            ]
            
            # Try to generate document-specific suggestions
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "lambda_mult": 0.3}
                )
                
                sample_docs = retriever.get_relevant_documents("main topics key concepts important information")
                
                if sample_docs:
                    # Create content sample
                    content_sample = "\n\n".join([doc.page_content for doc in sample_docs[:5]])
                    
                    suggestion_prompt = f"""
                    Based on this multi-document content, suggest 5 intelligent questions that would help users explore and understand the documents better.
                    
                    Focus on:
                    - Cross-document analysis
                    - Comparative questions
                    - Synthesis questions
                    - Key concept exploration
                    
                    Content sample:
                    {content_sample[:2000]}
                    
                    Provide only the questions, one per line:
                    """
                    
                    response = self.llm.invoke(suggestion_prompt)
                    suggested_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # Extract questions
                    suggested_questions = []
                    for line in suggested_text.split('\n'):
                        line = line.strip()
                        if line and ('?' in line or line.endswith('.')):
                            # Clean up the question
                            question = line.strip('- •').strip()
                            if question and len(question) > 10:
                                suggested_questions.append(question)
                    
                    if suggested_questions:
                        return base_suggestions + suggested_questions[:5]
            except Exception as e:
                print(f"Could not generate custom suggestions: {e}")
            
            return base_suggestions
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return [
                "What are the main topics covered in the documents?",
                "Compare the key points between documents",
                "What are the most important findings?",
                "Summarize the overall content"
            ]
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions"""
        return self.suggested_questions
    
    def _setup_advanced_features(self):
        """Initialize advanced features for multi-document analysis"""
        try:
            if hasattr(self, 'vectorstore') and hasattr(self, 'llm') and self.vectorstore and self.llm:
                self.advanced_analyzer = AdvancedDocumentAnalyzer(
                    self.vectorstore, 
                    self.llm
                )
                self.document_comparison = DocumentComparison(self.llm)
                self.smart_search = SmartSearch(self.vectorstore, self.llm)
                print("✅ Advanced multi-document features initialized")
            else:
                print("⚠️ Vectorstore or LLM not available for advanced features")
                self.advanced_analyzer = None
                self.document_comparison = None
                self.smart_search = None
        except Exception as e:
            print(f"⚠️ Could not initialize advanced features: {e}")
            self.advanced_analyzer = None
            self.document_comparison = None
            self.smart_search = None
    
    def setup_agent(self):
        """Setup the enhanced agent with multi-document tools"""
        try:
            print("Setting up enhanced multi-document agent...")
            
            # Enhanced tools for multi-document analysis
            tools = [
                Tool(
                    name="multi_document_search",
                    description="Search across all documents for specific information. Use for targeted queries about specific topics, facts, or concepts.",
                    func=self.multi_document_search_tool
                ),
                Tool(
                    name="cross_document_analysis",
                    description="Perform comprehensive analysis across multiple documents. Use for overview questions, comparisons, and synthesis.",
                    func=self.cross_document_analysis_tool
                ),
                Tool(
                    name="document_comparison",
                    description="Compare specific aspects between documents. Use when asked to compare, contrast, or find differences.",
                    func=self.document_comparison_tool
                ),
                Tool(
                    name="entity_analysis",
                    description="Analyze entities and their relationships across documents. Use for questions about people, organizations, locations, etc.",
                    func=self.entity_analysis_tool
                ),
                Tool(
                    name="theme_analysis",
                    description="Find common themes and topics across all documents. Use for thematic analysis questions.",
                    func=self.theme_analysis_tool
                ),
                Tool(
                    name="document_synthesis",
                    description="Synthesize information from all documents into unified insights. Use for synthesis and summary questions.",
                    func=self.document_synthesis_tool
                )
            ]
            
            # Get React prompt
            try:
                prompt = hub.pull("hwchase17/react")
            except:
                # Fallback prompt if hub is not available
                prompt = PromptTemplate(
                    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                    template="""
                    You are an expert multi-document analysis assistant with access to specialized tools.
                    
                    You have access to the following tools:
                    {tools}
                    
                    Use the following format:
                    
                    Question: the input question you must answer
                    Thought: you should always think about what to do
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question
                    
                    Question: {input}
                    Thought: {agent_scratchpad}
                    """
                )
            
            # Create agent
            agent = create_react_agent(self.llm, tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=5,
                handle_parsing_errors=True
            )
            
            print("✅ Enhanced multi-document agent setup complete!")
        except Exception as e:
            print(f"⚠️ Could not setup agent: {e}")
            self.agent_executor = None
    
    # Enhanced tool methods for multi-document analysis
    def multi_document_search_tool(self, query: str) -> str:
        """Enhanced search tool for multi-document queries"""
        try:
            retriever = self.get_retriever_by_type(self.search_type)
            docs = retriever.get_relevant_documents(query)
            
            # Group results by document
            doc_results = defaultdict(list)
            for doc in docs[:8]:
                doc_name = doc.metadata.get("document_name", "Unknown")
                doc_results[doc_name].append(doc.page_content)
            
            # Create comprehensive response
            response_parts = []
            for doc_name, contents in doc_results.items():
                response_parts.append(f"From {doc_name}:")
                response_parts.append("\n".join(contents[:2]))
                response_parts.append("")
            
            return "\n".join(response_parts)
        except Exception as e:
            return f"Error in multi-document search: {str(e)}"
    
    def cross_document_analysis_tool(self, query: str) -> str:
        """Tool for cross-document analysis"""
        try:
            # Determine analysis type based on query
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["compare", "contrast", "difference", "similar"]):
                result = self.compare_documents()
                return result.get("comparison", "No comparison available")
            
            elif any(word in query_lower for word in ["theme", "topic", "common"]):
                result = self.find_common_themes()
                if "themes" in result:
                    themes_text = "Common themes across documents:\n"
                    for theme in result["themes"][:5]:
                        themes_text += f"• {theme['name']} (in {len(theme['documents'])} documents)\n"
                    return themes_text
                return "No common themes found"
            
            elif any(word in query_lower for word in ["entity", "entities", "people", "organization"]):
                result = self.analyze_entity_overlap()
                if "entity_overlap" in result:
                    entities_text = "Cross-document entities:\n"
                    for entity, info in list(result["entity_overlap"].items())[:10]:
                        entities_text += f"• {entity} ({info['type']}) - appears in {len(info['documents'])} documents\n"
                    return entities_text
                return "No cross-document entities found"
            
            else:
                # General synthesis
                result = self.synthesize_documents()
                return result.get("analysis", "No synthesis available")
                
        except Exception as e:
            return f"Error in cross-document analysis: {str(e)}"
    
    def document_comparison_tool(self, query: str) -> str:
        """Tool for document comparison"""
        try:
            result = self.compare_documents()
            return result.get("comparison", "No comparison available")
        except Exception as e:
            return f"Error in document comparison: {str(e)}"
    
    def entity_analysis_tool(self, query: str) -> str:
        """Tool for entity analysis"""
        try:
            result = self.analyze_entity_overlap()
            if "entity_overlap" in result:
                analysis = "Entity Analysis Across Documents:\n\n"
                for entity, info in list(result["entity_overlap"].items())[:15]:
                    analysis += f"**{entity}** ({info['type']})\n"
                    analysis += f"  - Documents: {', '.join(info['documents'])}\n"
                    analysis += f"  - Frequency: {info['frequency']}\n\n"
                return analysis
            return "No entity overlap found"
        except Exception as e:
            return f"Error in entity analysis: {str(e)}"
    
    def theme_analysis_tool(self, query: str) -> str:
        """Tool for theme analysis"""
        try:
            result = self.find_common_themes()
            if "themes" in result:
                analysis = "Theme Analysis Across Documents:\n\n"
                for theme in result["themes"][:10]:
                    analysis += f"**{theme['name']}**\n"
                    analysis += f"  - Documents: {', '.join(theme['documents'])}\n"
                    analysis += f"  - Description: {theme['description']}\n\n"
                return analysis
            return "No common themes found"
        except Exception as e:
            return f"Error in theme analysis: {str(e)}"
    
    def document_synthesis_tool(self, query: str) -> str:
        """Tool for document synthesis"""
        try:
            result = self.synthesize_documents()
            return result.get("analysis", "No synthesis available")
        except Exception as e:
            return f"Error in document synthesis: {str(e)}"
    
    def ask_question(self, 
                    question: str, 
                    search_type: Optional[str] = None,
                    analysis_mode: Optional[str] = None,
                    response_style: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced question answering with multi-document support"""
        try:
            # Update settings if provided
            if search_type:
                self.search_type = search_type
            if analysis_mode:
                self.analysis_mode = analysis_mode
            if response_style:
                self.response_style = response_style
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "timestamp": str(datetime.now()),
                "search_type": self.search_type,
                "analysis_mode": self.analysis_mode
            })
            
            # Use agent if available, otherwise fallback to QA chain
            if self.agent_executor:
                try:
                    result = self.agent_executor.invoke({"input": question})
                    answer = result.get("output", "No answer generated")
                    mode = "Agent"
                except Exception as e:
                    print(f"Agent failed, falling back to QA chain: {e}")
                    result = self.qa_chain.invoke({"query": question})
                    answer = result.get("result", "No answer generated")
                    mode = "QA Chain"
            else:
                result = self.qa_chain.invoke({"query": question})
                answer = result.get("result", "No answer generated")
                mode = "QA Chain"
            
            # Extract source documents
            documents_used = []
            if isinstance(result, dict) and "source_documents" in result:
                for doc in result["source_documents"]:
                    doc_name = doc.metadata.get("document_name", "Unknown")
                    if doc_name not in documents_used:
                        documents_used.append(doc_name)
            
            return {
                "answer": answer,
                "mode": mode,
                "search_type": self.search_type,
                "analysis_mode": self.analysis_mode,
                "response_style": self.response_style,
                "documents_used": documents_used,
                "total_documents": len(self.document_paths)
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "mode": "Error",
                "search_type": self.search_type,
                "documents_used": []
            }

def main():
    """Test the enhanced chatbot"""
    # Example usage
    document_paths = ["example1.pdf", "example2.pdf"]  # Replace with actual paths
    
    try:
        chatbot = EnhancedAgenticDocumentQAChatbot(
            document_paths=document_paths,
            embedding_model="multilingual-e5-large",
            search_type="hybrid",
            force_recreate=True
        )
        
        print("Enhanced Multi-Document Chatbot initialized successfully!")
        
        # Test questions
        test_questions = [
            "What are the main themes across all documents?",
            "Compare the key findings between documents",
            "What entities appear in multiple documents?"
        ]
        
        for question in test_questions:
            print(f"\nQ: {question}")
            response = chatbot.ask_question(question)
            print(f"A: {response['answer']}")
            print(f"Mode: {response['mode']}, Documents used: {response['documents_used']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()