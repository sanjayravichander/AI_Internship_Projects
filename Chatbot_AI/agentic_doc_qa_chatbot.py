import os
import hashlib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub
from typing import Dict, List, Any
import json
from advanced_features import AdvancedDocumentAnalyzer, DocumentComparison, SmartSearch

# Load environment variables from .env in project root or as specified by ENV_PATH
import os
#os.environ["TRANSFORMERS_NO_TF"] = "1"
#os.environ["USE_TF"] = "0"
from dotenv import load_dotenv
dotenv_path = os.getenv("ENV_PATH", os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=dotenv_path)

class AgenticDocumentQAChatbot:
    def __init__(self, pdf_path, vectorstore_base_path=None, force_recreate=False):
        if vectorstore_base_path is None:
            # Default to vectorstore in project root, OS-agnostic
            vectorstore_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore"))
        """
        Initialize the Agentic Document Q&A Chatbot
        
        Args:
            pdf_path (str): Path to the PDF document
            vectorstore_base_path (str): Base path for vector store directory
            force_recreate (bool): Force recreation of vector store for new documents
        """
        self.pdf_path = pdf_path
        self.vectorstore_base_path = vectorstore_base_path
        self.force_recreate = force_recreate
        self.vectorstore = None
        self.qa_chain = None
        self.agent_executor = None
        self.conversation_history = []
        self.document_metadata = {}  # Store document info for better context
        
        # Generate unique vector store path based on PDF content/name
        self.vectorstore_path = self._generate_vectorstore_path()
        
        # Initialize Groq API key
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Please set your GROQ_API_KEY in the .env file")
        
        # Clear existing vector store if force_recreate is True
        if self.force_recreate:
            self._clear_existing_vectorstore()
        
        # Initialize the chatbot
        self.setup_chatbot()
        self.setup_agent()
        
        # Initialize advanced features
        self.advanced_analyzer = None
        self.document_comparison = None
        self.smart_search = None
        self._setup_advanced_features()
        
        # Generate suggested questions based on document content
        self.suggested_questions = self.generate_suggested_questions()
    
    def _generate_vectorstore_path(self):
        """Generate a unique vector store path based on PDF file"""
        # Create a hash of the PDF file path and name for uniqueness
        pdf_identifier = f"{os.path.basename(self.pdf_path)}_{os.path.getsize(self.pdf_path) if os.path.exists(self.pdf_path) else 'temp'}"
        pdf_hash = hashlib.md5(pdf_identifier.encode()).hexdigest()[:8]
        
        # Create unique path
        vectorstore_path = os.path.join(self.vectorstore_base_path, f"faiss_index_{pdf_hash}")
        
        print(f"📁 Vector store path: {vectorstore_path}")
        return vectorstore_path
    
    def _clear_existing_vectorstore(self):
        """Clear existing vector store directory"""
        import shutil
        try:
            if os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)
                print(f"🗑️ Cleared existing vector store at: {self.vectorstore_path}")
            
            # Also clear the base vectorstore directory if it exists
            if os.path.exists(self.vectorstore_base_path):
                for item in os.listdir(self.vectorstore_base_path):
                    item_path = os.path.join(self.vectorstore_base_path, item)
                    if os.path.isdir(item_path) and item.startswith("faiss_index_"):
                        shutil.rmtree(item_path)
                        print(f"🗑️ Cleared old vector store: {item}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear existing vector store: {e}")
    
    def load_and_process_document(self):
        """Load PDF document and split into chunks with enhanced processing"""
        print("Loading and processing document...")
        
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Store document metadata for better context
        self.document_metadata = {
            "total_pages": len(pages),
            "document_name": os.path.basename(self.pdf_path),
            "total_content_length": sum(len(page.page_content) for page in pages)
        }
        
        print(f"📄 Document loaded: {self.document_metadata['total_pages']} pages, {self.document_metadata['total_content_length']} characters")
        
        # Enhanced document splitting for better comprehensive analysis
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Increased for more context per chunk
            chunk_overlap=400,  # Higher overlap for better continuity
            separators=["\n\n", "\n", ". ", " ", ""],  # Better separation logic
            length_function=len
        )
        docs = splitter.split_documents(pages)
        
        # Add enhanced metadata to each chunk
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": i,
                "total_chunks": len(docs),
                "document_name": self.document_metadata["document_name"],
                "total_pages": self.document_metadata["total_pages"]
            })
        
        print(f"📊 Document split into {len(docs)} enhanced chunks")
        return docs
    
    def create_vectorstore(self, docs):
        """Create embeddings and vector store"""
        print("Creating embeddings and vector store...")
        
        # Try Jina embeddings first, fallback to BGE if it fails
        try:
            print("🔄 Attempting to use Jina AI embeddings...")
            embed_model = HuggingFaceEmbeddings(
                model_name="jinaai/jina-embeddings-v3"
            )
            print("✅ Jina AI embeddings loaded successfully!")
        except Exception as e:
            print(f"⚠️ Jina AI embeddings failed: {e}")
            print("🔄 Falling back to BGE embeddings...")
            embed_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )
            print("✅ BGE embeddings loaded successfully!")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embed_model)
        
        # Save vector store locally
        vectorstore.save_local(self.vectorstore_path)
        print("Vector store created and saved successfully!")
        
        return vectorstore
    
    def load_existing_vectorstore(self):
        """Load existing vector store if available"""
        try:
            # Try Jina embeddings first, fallback to BGE if it fails
            try:
                embed_model = HuggingFaceEmbeddings(
                    model_name="jinaai/jina-embeddings-v3"
                )
            except Exception:
                embed_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5"
                )
            
            vectorstore = FAISS.load_local(
                self.vectorstore_path, 
                embed_model,
                allow_dangerous_deserialization=True
            )
            print("Existing vector store loaded successfully!")
            return vectorstore
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return None
    
    def setup_qa_chain(self):
        """Setup the Q&A chain with Groq"""
        print("Setting up Q&A chain...")
        
        # Initialize Groq LLM optimized for speed and focused responses
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,  # Lower temperature for more focused answers
            max_tokens=500  # Reduced for faster, more concise responses
        )
        
        # Create enhanced prompt template with document context
        prompt_template = f"""
        You are an expert AI assistant analyzing a document titled "{self.document_metadata.get('document_name', 'Unknown Document')}".
        This document has {self.document_metadata.get('total_pages', 'unknown')} pages and contains comprehensive information.
        
        Your task is to provide accurate, detailed answers based on the provided context.
        
        IMPORTANT GUIDELINES:
        1. Use ALL the provided context to give comprehensive answers
        2. When asked about "what the document is about", provide a detailed overview covering main topics, themes, and key points
        3. For counting or listing questions, be thorough and systematic
        4. If the context seems incomplete for a comprehensive answer, mention that you're providing information based on the available sections
        5. Always be specific and cite relevant details from the context
        6. If you don't know something based on the context, clearly state that
        
        Context from the document:
        {{context}}
        
        Question: {{question}}
        
        Detailed Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain with focused retrieval settings
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",  # Use similarity for more focused results
                search_kwargs={
                    "k": 6,  # Reduced for faster, more focused responses
                }
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("Q&A chain setup complete!")
        return llm
    
    def setup_chatbot(self):
        """Setup the complete chatbot pipeline"""
        # Check if we should force recreate or try to load existing
        if self.force_recreate:
            print("🔄 Force recreating vector store for new document...")
            docs = self.load_and_process_document()
            self.vectorstore = self.create_vectorstore(docs)
        else:
            # Try to load existing vector store first
            self.vectorstore = self.load_existing_vectorstore()
            
            # If no existing vector store, create new one
            if self.vectorstore is None:
                docs = self.load_and_process_document()
                self.vectorstore = self.create_vectorstore(docs)
        
        # Setup Q&A chain and get LLM
        self.llm = self.setup_qa_chain()
    
    def _setup_advanced_features(self):
        """Initialize advanced features after basic setup"""
        try:
            if hasattr(self, 'vectorstore') and hasattr(self, 'llm') and self.vectorstore and self.llm:
                self.advanced_analyzer = AdvancedDocumentAnalyzer(
                    self.vectorstore, 
                    self.llm
                )
                self.document_comparison = DocumentComparison(self.llm)
                self.smart_search = SmartSearch(self.vectorstore, self.llm)
                print("✅ Advanced features initialized")
            else:
                print("⚠️ Vectorstore or LLM not available for advanced features")
                self.advanced_analyzer = None
                self.document_comparison = None
                self.smart_search = None
        except Exception as e:
            print(f"⚠️ Could not initialize advanced features: {e}")
            # Set to None to prevent further errors
            self.advanced_analyzer = None
            self.document_comparison = None
            self.smart_search = None
    
    def document_search_tool(self, query: str) -> str:
        """Tool for searching the document with focused results"""
        try:
            # Use focused retrieval for specific questions
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Even more focused for specific questions
            )
            docs = retriever.get_relevant_documents(query)
            
            # Combine relevant content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create focused prompt
            focused_prompt = f"""
            Based on the specific context provided, answer the question directly and concisely.
            Focus only on the relevant information that answers the question.
            Do not include unrelated information.
            
            Context:
            {context}
            
            Question: {query}
            
            Focused Answer:"""
            
            response = self.llm.invoke(focused_prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return answer
        except Exception as e:
            return f"Error searching document: {str(e)}"
    
    def comprehensive_analysis_tool(self, query: str) -> str:
        """Tool for comprehensive document analysis (counting, listing, overview questions)"""
        try:
            # Check if this is a document overview question
            overview_keywords = ["what is", "about", "overview", "summary", "document", "content", "topic", "theme"]
            is_overview = any(keyword in query.lower() for keyword in overview_keywords)
            
            if is_overview:
                # For overview questions, get a broader sample of the document
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 20, "lambda_mult": 0.5}  # More diverse content
                )
            else:
                # For specific analysis, use targeted retrieval
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": 15}
                )
            
            docs = retriever.get_relevant_documents(query)
            
            # Combine all retrieved content
            combined_content = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a specialized prompt for comprehensive analysis
            comprehensive_prompt = f"""
            You are analyzing a document titled "{self.document_metadata.get('document_name', 'Unknown Document')}" 
            which has {self.document_metadata.get('total_pages', 'unknown')} pages.
            
            Use ALL the provided context to answer the question comprehensively.
            
            Context from multiple document sections:
            {combined_content}
            
            Question: {query}
            
            Instructions:
            - If asked about what the document is about, provide a comprehensive overview covering:
              * Main topics and themes
              * Key concepts and ideas
              * Purpose and scope of the document
              * Important findings or conclusions
            - If asked to count items, carefully go through ALL sections and count every occurrence
            - If asked to list items, provide a complete list from all sections
            - Be thorough and systematic in your analysis
            - Use specific details and examples from the context
            - If the context seems incomplete, mention that you're providing information based on the available sections
            
            Comprehensive Answer: """
            
            # Use the LLM directly for comprehensive analysis
            response = self.llm.invoke(comprehensive_prompt)
            
            # Extract the response content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            result = f"{answer}\n\n"
            result += f"📊 Analysis based on {len(docs)} document sections from {self.document_metadata.get('total_pages', 'unknown')} total pages"
            
            return result
        except Exception as e:
            return f"Error in comprehensive analysis: {str(e)}"
    
    def summarize_section_tool(self, section_topic: str) -> str:
        """Tool for summarizing specific sections"""
        query = f"Summarize the section about {section_topic}"
        return self.document_search_tool(query)
    
    def compare_concepts_tool(self, concepts: str) -> str:
        """Tool for comparing two concepts"""
        # Parse the input to extract two concepts
        if ',' in concepts:
            concept1, concept2 = [c.strip() for c in concepts.split(',', 1)]
        else:
            # If no comma, assume it's two concepts separated by 'and'
            parts = concepts.split(' and ')
            if len(parts) >= 2:
                concept1, concept2 = parts[0].strip(), parts[1].strip()
            else:
                return "Please provide two concepts separated by comma or 'and' (e.g., 'encoder, decoder' or 'encoder and decoder')"
        
        query = f"Compare and contrast {concept1} and {concept2}"
        return self.document_search_tool(query)
    
    def find_definitions_tool(self, term: str) -> str:
        """Tool for finding definitions"""
        query = f"What is the definition of {term}? How is {term} defined?"
        return self.document_search_tool(query)
    
    def conversation_memory_tool(self, action: str) -> str:
        """Tool for managing conversation memory"""
        if action == "recall":
            if self.conversation_history:
                recent = self.conversation_history[-3:]  # Last 3 exchanges
                return f"Recent conversation: {json.dumps(recent, indent=2)}"
            return "No previous conversation history."
        elif action == "clear":
            self.conversation_history = []
            return "Conversation history cleared."
        return "Invalid action. Use 'recall' or 'clear'."
    
    def document_overview_tool(self, query: str) -> str:
        """Specialized tool for document overview and 'what is this document about' questions"""
        try:
            # Get a diverse sample of the document for overview (optimized for speed)
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 12,  # Reduced for faster processing
                    "lambda_mult": 0.4,  # Balance between diversity and speed
                    "fetch_k": 20  # Reduced candidates for speed
                }
            )
            
            # Use a broad query to get diverse content
            overview_query = f"overview summary main topics themes content purpose {query}"
            docs = retriever.get_relevant_documents(overview_query)
            
            # Combine content from different parts of the document
            combined_content = "\n\n".join([doc.page_content for doc in docs])
            
            # Create specialized overview prompt
            overview_prompt = f"""
            You are providing a comprehensive overview of the document "{self.document_metadata.get('document_name', 'Unknown Document')}" 
            which has {self.document_metadata.get('total_pages', 'unknown')} pages.
            
            Based on the diverse content sections provided below, give a detailed overview that covers:
            
            1. MAIN PURPOSE: What is the primary purpose or objective of this document?
            2. KEY TOPICS: What are the main topics, themes, or subjects covered?
            3. CONTENT STRUCTURE: How is the information organized or structured?
            4. KEY CONCEPTS: What are the most important concepts, ideas, or findings?
            5. TARGET AUDIENCE: Who appears to be the intended audience?
            6. SCOPE: What is the scope and breadth of coverage?
            
            Content from multiple document sections:
            {combined_content}
            
            User Question: {query}
            
            Provide a comprehensive, well-structured overview that gives the user a clear understanding of what this document is all about:
            """
            
            # Use the LLM directly for overview generation
            response = self.llm.invoke(overview_prompt)
            
            # Extract the response content
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            result = f"{answer}\n\n"
            result += f"📊 Overview based on analysis of {len(docs)} diverse sections from {self.document_metadata.get('total_pages', 'unknown')} total pages"
            
            return result
        except Exception as e:
            return f"Error generating document overview: {str(e)}"
    
    def generate_suggested_questions(self):
        """Generate intelligent question suggestions based on document content"""
        try:
            print("🤔 Generating suggested questions based on document analysis...")
            
            # Get diverse content samples for analysis
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 8, "lambda_mult": 0.3}
            )
            
            # Use broad queries to get diverse content
            sample_docs = retriever.get_relevant_documents("main topics concepts key points important information")
            
            # Combine content for analysis
            content_sample = "\n\n".join([doc.page_content for doc in sample_docs])
            
            # Create prompt for question generation
            question_prompt = f"""
            Based on the document content below, generate 6 intelligent questions that users would likely want to ask about this document.
            
            The questions should be:
            1. Relevant to the main topics and themes
            2. Specific and actionable
            3. Cover different aspects of the document
            4. Be natural and conversational
            
            Document content sample:
            {content_sample[:3000]}...
            
            Generate exactly 6 questions in this format:
            1. [Question about main topic/purpose]
            2. [Question about key concepts]
            3. [Question about specific details]
            4. [Question about methodology/approach]
            5. [Question about outcomes/results]
            6. [Question about practical applications]
            
            Questions:"""
            
            response = self.llm.invoke(question_prompt)
            questions_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse questions from response
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Clean up the question
                    question = line.split('.', 1)[-1].strip()
                    question = question.replace('[', '').replace(']', '')
                    if question and len(question) > 10:
                        questions.append(question)
            
            # Fallback questions if generation fails
            if len(questions) < 3:
                questions = [
                    "What is this document about?",
                    "What are the main topics covered?",
                    "What are the key concepts explained?",
                    "How is the information structured?",
                    "What are the important findings?",
                    "What practical applications are discussed?"
                ]
            
            print(f"✅ Generated {len(questions)} suggested questions")
            return questions[:6]  # Return max 6 questions
            
        except Exception as e:
            print(f"⚠️ Error generating suggested questions: {e}")
            # Return default questions
            return [
                "What is this document about?",
                "What are the main topics covered?",
                "What are the key concepts explained?",
                "How is the information structured?",
                "What are the important findings?",
                "What practical applications are discussed?"
            ]
    
    def get_suggested_questions(self):
        """Get the suggested questions for the UI"""
        return getattr(self, 'suggested_questions', [])
    
    # Advanced Features Methods
    def analyze_document_complexity(self):
        """Analyze document complexity and readability"""
        if not self.advanced_analyzer:
            return {"error": "Advanced analyzer not available"}
        
        try:
            # Get full document text
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 50})
            docs = retriever.get_relevant_documents("document content text analysis")
            full_text = "\n\n".join([doc.page_content for doc in docs])
            
            return self.advanced_analyzer.analyze_document_complexity(full_text)
        except Exception as e:
            return {"error": f"Error analyzing complexity: {str(e)}"}
    
    def extract_key_entities(self):
        """Extract key entities from the document"""
        if not self.advanced_analyzer:
            return {"error": "Advanced analyzer not available"}
        
        try:
            # Get document text
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 30})
            docs = retriever.get_relevant_documents("entities people organizations locations")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            return self.advanced_analyzer.extract_key_entities(text)
        except Exception as e:
            return {"error": f"Error extracting entities: {str(e)}"}
    
    def classify_document_type(self):
        """Classify the document type"""
        if not self.advanced_analyzer:
            return {"error": "Advanced analyzer not available"}
        
        try:
            # Get document sample
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents("document structure content type")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            return self.advanced_analyzer.classify_document_type(text)
        except Exception as e:
            return {"error": f"Error classifying document: {str(e)}"}
    
    def generate_executive_summary(self, summary_type: str = "executive"):
        """Generate different types of summaries"""
        if not self.advanced_analyzer:
            return "Advanced analyzer not available"
        
        try:
            # Get comprehensive document content
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 20, "lambda_mult": 0.5}
            )
            docs = retriever.get_relevant_documents("summary overview main points key findings")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            return self.advanced_analyzer.generate_executive_summary(text, summary_type)
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def perform_semantic_search(self, query: str, search_type: str = "comprehensive"):
        """Perform advanced semantic search"""
        if not self.smart_search:
            return {"error": "Smart search not available"}
        
        try:
            return self.smart_search.semantic_search(query, search_type)
        except Exception as e:
            return {"error": f"Error in semantic search: {str(e)}"}
    
    def find_citations_and_references(self):
        """Find citations and references in the document"""
        if not self.smart_search:
            return {"error": "Smart search not available"}
        
        try:
            # Get document text
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 50})
            docs = retriever.get_relevant_documents("references citations bibliography")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            return self.smart_search.find_citations_and_references(text)
        except Exception as e:
            return {"error": f"Error finding references: {str(e)}"}
    
    def generate_knowledge_graph(self):
        """Generate knowledge graph from document"""
        if not self.advanced_analyzer:
            return {"error": "Advanced analyzer not available"}
        
        try:
            # Get document text for graph generation
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 30})
            docs = retriever.get_relevant_documents("entities relationships concepts")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            result = self.advanced_analyzer.generate_knowledge_graph(text)
            
            # Ensure result is serializable
            if isinstance(result, dict):
                # Convert any non-serializable objects to strings or remove them
                serializable_result = {}
                for k, v in result.items():
                    if isinstance(v, (str, int, float, bool, list, dict)):
                        serializable_result[k] = v
                    else:
                        serializable_result[k] = str(v)
                return serializable_result
            else:
                return {"error": "Invalid result format"}
        except Exception as e:
            return {"error": f"Error generating knowledge graph: {str(e)}"}
    
    def _safe_analyze_document_complexity(self):
        """Safe wrapper for document complexity analysis"""
        try:
            if not self.advanced_analyzer:
                return {"error": "Advanced analyzer not available"}
            
            # Get sample text for analysis
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents("document content analysis")
            text = "\n\n".join([doc.page_content for doc in docs[:5]])  # Limit text
            
            result = self.advanced_analyzer.analyze_document_complexity(text)
            
            # Ensure result is serializable
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict))}
            else:
                return {"error": "Invalid result format"}
        except Exception as e:
            return {"error": f"Error analyzing complexity: {str(e)}"}
    
    def _safe_classify_document_type(self):
        """Safe wrapper for document classification"""
        try:
            if not self.advanced_analyzer:
                return {"error": "Advanced analyzer not available"}
            
            # Get sample text for classification
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents("document type classification")
            text = "\n\n".join([doc.page_content for doc in docs[:3]])  # Limit text
            
            result = self.advanced_analyzer.classify_document_type(text)
            
            # Ensure result is serializable
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict))}
            else:
                return {"error": "Invalid result format"}
        except Exception as e:
            return {"error": f"Error classifying document: {str(e)}"}
    
    def _safe_extract_key_entities(self):
        """Safe wrapper for key entity extraction"""
        try:
            if not self.advanced_analyzer:
                return {"error": "Advanced analyzer not available"}
            
            # Get sample text for entity extraction
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents("entities names organizations")
            text = "\n\n".join([doc.page_content for doc in docs[:5]])  # Limit text
            
            result = self.advanced_analyzer.extract_key_entities(text)
            
            # Ensure result is serializable
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict))}
            else:
                return {"error": "Invalid result format"}
        except Exception as e:
            return {"error": f"Error extracting entities: {str(e)}"}
    
    def _safe_find_citations_and_references(self):
        """Safe wrapper for finding citations and references"""
        try:
            if not self.smart_search:
                return {"error": "Smart search not available"}
            
            # Get document text
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 20})
            docs = retriever.get_relevant_documents("references citations bibliography")
            text = "\n\n".join([doc.page_content for doc in docs])
            
            result = self.smart_search.find_citations_and_references(text)
            
            # Ensure result is serializable
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict))}
            else:
                return {"error": "Invalid result format"}
        except Exception as e:
            return {"error": f"Error finding references: {str(e)}"}
    
    def get_document_insights(self):
        """Get comprehensive document insights"""
        try:
            insights = {
                "basic_info": self.document_metadata,
                "complexity_analysis": self._safe_analyze_document_complexity(),
                "document_classification": self._safe_classify_document_type(),
                "key_entities": self._safe_extract_key_entities(),
                "citations_references": self._safe_find_citations_and_references()
            }
            
            return insights
        except Exception as e:
            print(f"Error getting document insights: {e}")
            return {
                "basic_info": self.document_metadata,
                "complexity_analysis": {"error": "Analysis unavailable"},
                "document_classification": {"error": "Classification unavailable"},
                "key_entities": {"error": "Entity extraction unavailable"},
                "citations_references": {"error": "Citation analysis unavailable"}
            }
    
    def setup_agent(self):
        """Setup the agentic system with tools"""
        print("Setting up agentic system...")
        
        # Define tools with enhanced descriptions
        tools = [
            Tool(
                name="DocumentOverview",
                func=self.document_overview_tool,
                description="Provides comprehensive overview of the document including main purpose, key topics, structure, and scope. Use for questions about what the document is about or document summaries."
            ),
            Tool(
                name="ComprehensiveAnalysis",
                func=self.comprehensive_analysis_tool,
                description="Performs detailed analysis for counting, listing, and enumeration tasks. Use for 'How many...', 'List all...', 'What are all...' questions."
            ),
            Tool(
                name="DocumentSearch",
                func=self.document_search_tool,
                description="Searches for specific information in the document. Use for factual questions about particular topics or concepts."
            ),
            Tool(
                name="SummarizeSection", 
                func=self.summarize_section_tool,
                description="Summarizes specific sections or topics from the document. Input should be the topic name."
            ),
            Tool(
                name="CompareConcepts",
                func=self.compare_concepts_tool,
                description="Compares two concepts from the document. Input should be 'concept1,concept2'."
            ),
            Tool(
                name="FindDefinitions",
                func=self.find_definitions_tool,
                description="Finds definitions of terms or concepts. Input should be the term to define."
            ),
            Tool(
                name="ConversationMemory",
                func=self.conversation_memory_tool,
                description="Manages conversation history. Input should be 'recall' or 'clear'."
            )
        ]
        
        # Create a custom prompt that ensures single tool usage and proper final answers
        agent_prompt = PromptTemplate.from_template("""
        You are an expert AI research assistant analyzing documents.
        You have access to specialized tools to provide comprehensive and accurate answers.
        
        CRITICAL RULES:
        1. Use ONLY ONE tool per question
        2. After using a tool, IMMEDIATELY provide your Final Answer based on the tool's result
        3. Do NOT use multiple tools or additional tools after getting a result
        4. Use the exact tool name without any parameters in the Action field
        
        TOOL SELECTION:
        - For "What is this document about?" or overview questions → use DocumentOverview
        - For counting or listing → use ComprehensiveAnalysis  
        - For specific facts → use DocumentSearch
        - For definitions → use FindDefinitions
        - For comparisons → use CompareConcepts
        - For section summaries → use SummarizeSection
        
        Available tools:
        {tools}
        
        Use the following format EXACTLY:
        
        Question: the input question you must answer
        Thought: I will use [TOOL_NAME] to answer this question because [reason]
        Action: [TOOL_NAME]
        Action Input: [the user's question or relevant input]
        Observation: the result of the action
        Thought: I now have the complete answer from the tool
        Final Answer: [Provide the tool's result as the final answer]
        
        Question: {input}
        Thought:{agent_scratchpad}
        """)
        
        # Create agent
        try:
            # Debug: Print tool names
            tool_names = [tool.name for tool in tools]
            print(f"🔧 Available tools: {tool_names}")
            
            # Use the standard hub prompt which works reliably
            try:
                agent_prompt = hub.pull("hwchase17/react")
                print("✅ Using standard hub prompt")
            except:
                print("⚠️ Hub not available, using basic prompt")
                agent_prompt = PromptTemplate.from_template("""
                Answer the following questions as best you can. You have access to the following tools:

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

                Begin!

                Question: {input}
                Thought:{agent_scratchpad}
                """)
            
            agent = create_react_agent(self.llm, tools, agent_prompt)
            self.agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=False,  # Reduce verbosity for speed
                max_iterations=1,  # Force single tool usage for speed
                max_execution_time=30,  # Reduced time limit
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            print("Agentic system setup complete!")
            print(f"✅ Agent created with tools: {tool_names}")
        except Exception as e:
            print(f"Could not setup agent (falling back to basic mode): {e}")
            self.agent_executor = None
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question using the agentic system"""
        # Add to conversation history
        self.conversation_history.append({"type": "question", "content": question})
        
        # Check if this is an overview question and handle directly if agent fails
        overview_keywords = ["what is this document about", "what does this document cover", "tell me about this document", "what is uploaded document all about", "document overview", "summarize this document"]
        is_overview_question = any(keyword in question.lower() for keyword in overview_keywords)
        
        try:
            if self.agent_executor:
                # Use agentic approach
                response = self.agent_executor.invoke({"input": question})
                answer = response.get("output", "")
                
                # Extract tools used from intermediate steps
                tools_used = []
                tool_results = []
                if "intermediate_steps" in response:
                    for step in response["intermediate_steps"]:
                        if len(step) >= 2:
                            if hasattr(step[0], 'tool'):
                                tools_used.append(step[0].tool)
                            tool_results.append(step[1])  # Tool result
                
                # If no proper answer but we have tool results, use the tool result
                if not answer or "Agent stopped due to iteration limit" in answer or answer.strip() == "" or "time limit" in answer.lower():
                    if tool_results:
                        # Use the first tool result (which should be the main answer)
                        answer = tool_results[0] if tool_results else ""
                    else:
                        # Special handling for overview questions
                        if is_overview_question:
                            try:
                                answer = self.document_overview_tool(question)
                                self.conversation_history.append({"type": "answer", "content": answer})
                                return {
                                    "answer": answer,
                                    "mode": "direct_overview",
                                    "tools_used": ["DocumentOverview"],
                                    "sources": "Direct tool call"
                                }
                            except Exception:
                                pass
                        
                        # Fallback to basic Q&A if agent fails
                        try:
                            response = self.qa_chain.invoke({"query": question})
                            answer = response["result"]
                            sources = response.get("source_documents", [])
                            
                            self.conversation_history.append({"type": "answer", "content": answer})
                            
                            return {
                                "answer": answer,
                                "mode": "basic",
                                "sources": len(sources),
                                "note": "Switched to basic mode due to agent failure"
                            }
                        except Exception:
                            answer = f"I apologize, but I'm having trouble processing your question. Please try rephrasing it or ask something else."
                
                # Add to conversation history
                self.conversation_history.append({"type": "answer", "content": answer})
                
                return {
                    "answer": answer,
                    "mode": "agentic",
                    "tools_used": tools_used,
                    "sources": f"{len(tools_used)} tools used"
                }
            else:
                # Fallback to basic Q&A
                response = self.qa_chain.invoke({"query": question})
                answer = response["result"]
                sources = response.get("source_documents", [])
                
                # Add to conversation history
                self.conversation_history.append({"type": "answer", "content": answer})
                
                return {
                    "answer": answer,
                    "mode": "basic",
                    "sources": len(sources),
                    "source_pages": [doc.metadata.get("page", "Unknown") for doc in sources[:2]]
                }
        
        except Exception as e:
            # If agent fails, try basic Q&A as fallback
            if "iteration limit" in str(e).lower() or "time limit" in str(e).lower():
                try:
                    print("⚠️ Agent timeout, falling back to basic Q&A...")
                    response = self.qa_chain.invoke({"query": question})
                    answer = response["result"]
                    sources = response.get("source_documents", [])
                    
                    self.conversation_history.append({"type": "answer", "content": answer})
                    
                    return {
                        "answer": answer,
                        "mode": "basic",
                        "sources": len(sources),
                        "note": "Switched to basic mode due to agent timeout"
                    }
                except Exception as fallback_error:
                    error_msg = f"Error in both agent and basic mode: {str(fallback_error)}"
                    self.conversation_history.append({"type": "error", "content": error_msg})
                    return {"answer": error_msg, "mode": "error"}
            else:
                error_msg = f"Error processing question: {str(e)}"
                self.conversation_history.append({"type": "error", "content": error_msg})
                return {"answer": error_msg, "mode": "error"}
    
    def chat_loop(self):
        """Interactive chat loop with agentic capabilities"""
        print("\n" + "="*70)
        print("🤖 Agentic Document Q&A Chatbot is ready!")
        print("📄 Loaded document:", os.path.basename(self.pdf_path))
        print("🧠 Agentic AI:", "Enabled" if self.agent_executor else "Disabled (Basic mode)")
        print("💡 Ask questions, request summaries, compare concepts, or find definitions")
        print("🔧 Special commands: 'memory recall', 'memory clear'")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("="*70 + "\n")
        
        while True:
            # Get user input
            user_question = input("👤 You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("🤖 Chatbot: Goodbye! Thanks for using the Agentic Document Q&A Chatbot!")
                break
            
            # Skip empty questions
            if not user_question:
                continue
            
            # Get chatbot response
            print("🤖 Chatbot: ", end="", flush=True)
            response = self.ask_question(user_question)
            
            print(response["answer"])
            
            # Show additional info
            if response.get("mode") == "agentic":
                print(f"🧠 (Agentic AI used multiple reasoning steps)")
            elif response.get("mode") == "basic":
                if response.get("sources", 0) > 0:
                    print(f"📚 (Based on {response['sources']} relevant document sections)")
            
            print()  # Add blank line for readability

def main():
    """Main function to run the agentic chatbot"""
    import argparse
    parser = argparse.ArgumentParser(description="Run Agentic Document QA Chatbot")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to analyze")
    args = parser.parse_args()
    pdf_path = args.pdf_path
    # Check if PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable with the correct path to your PDF file.")
        return
    
    try:
        # Initialize agentic chatbot
        chatbot = AgenticDocumentQAChatbot(pdf_path)
        
        # Start interactive chat
        chatbot.chat_loop()
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please create a .env file with your GROQ_API_KEY")
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")

if __name__ == "__main__":
    main()