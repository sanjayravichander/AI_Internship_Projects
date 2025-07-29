#!/usr/bin/env python3
"""
Advanced Features Module for Elite Document Q&A Chatbot
This module contains enterprise-grade features that showcase advanced AI/ML capabilities
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from langchain_groq import ChatGroq

class AdvancedDocumentAnalyzer:
    """Advanced document analysis capabilities"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.nlp = None
        self._load_nlp_model()
    
    def _load_nlp_model(self):
        """Load spaCy model for advanced NLP"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_document_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze document complexity and readability"""
        try:
            # Basic readability metrics
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            
            # Text statistics
            words = text.split()
            sentences = text.split('.')
            paragraphs = text.split('\n\n')
            
            # Vocabulary complexity
            unique_words = len(set(words))
            vocabulary_richness = unique_words / len(words) if words else 0
            
            # Average word and sentence length
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = np.mean([len(sent.split()) for sent in sentences if sent.strip()]) if sentences else 0
            
            # Complexity classification
            if flesch_score >= 90:
                complexity_level = "Very Easy"
            elif flesch_score >= 80:
                complexity_level = "Easy"
            elif flesch_score >= 70:
                complexity_level = "Fairly Easy"
            elif flesch_score >= 60:
                complexity_level = "Standard"
            elif flesch_score >= 50:
                complexity_level = "Fairly Difficult"
            elif flesch_score >= 30:
                complexity_level = "Difficult"
            else:
                complexity_level = "Very Difficult"
            
            return {
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_kincaid_grade": round(fk_grade, 2),
                "complexity_level": complexity_level,
                "total_words": len(words),
                "unique_words": unique_words,
                "vocabulary_richness": round(vocabulary_richness, 3),
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "total_sentences": len([s for s in sentences if s.strip()]),
                "total_paragraphs": len([p for p in paragraphs if p.strip()]),
                "estimated_reading_time": round(len(words) / 200, 1)  # 200 WPM average
            }
        except Exception as e:
            return {"error": f"Error analyzing complexity: {str(e)}"}
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities using NLP"""
        if not self.nlp:
            return {"error": "NLP model not available"}
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text size for processing
            
            entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],  # Geopolitical entities
                "DATE": [],
                "MONEY": [],
                "PRODUCT": [],
                "EVENT": [],
                "LAW": [],
                "LANGUAGE": []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            # Remove duplicates and sort
            for key in entities:
                entities[key] = sorted(list(set(entities[key])))
            
            return entities
        except Exception as e:
            return {"error": f"Error extracting entities: {str(e)}"}
    
    def perform_topic_modeling(self, documents: List[str], n_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling using LDA"""
        try:
            if len(documents) < 2:
                return {"error": "Need at least 2 documents for topic modeling"}
            
            # Vectorize documents
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(documents)),
                random_state=42,
                max_iter=50
            )
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weight": topic[top_words_idx].tolist()
                })
            
            return {
                "topics": topics,
                "n_topics": len(topics),
                "perplexity": lda.perplexity(doc_term_matrix)
            }
        except Exception as e:
            return {"error": f"Error in topic modeling: {str(e)}"}
    
    def generate_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """Generate a knowledge graph from document entities"""
        if not self.nlp:
            return {"error": "NLP model not available"}
        
        try:
            doc = self.nlp(text[:500000])  # Limit for processing
            
            # Extract entities and their relationships
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes (entities)
            for entity, label in entities:
                G.add_node(entity, type=label)
            
            # Add edges (co-occurrence in sentences)
            for sent in doc.sents:
                sent_entities = [ent.text for ent in sent.ents]
                # Connect entities that appear in the same sentence
                for i, ent1 in enumerate(sent_entities):
                    for ent2 in sent_entities[i+1:]:
                        if G.has_node(ent1) and G.has_node(ent2):
                            if G.has_edge(ent1, ent2):
                                G[ent1][ent2]['weight'] += 1
                            else:
                                G.add_edge(ent1, ent2, weight=1)
            
            # Convert to format suitable for visualization
            nodes = []
            edges = []
            
            for node in G.nodes():
                nodes.append({
                    "id": node,
                    "label": node,
                    "type": G.nodes[node].get('type', 'UNKNOWN'),
                    "size": G.degree(node)
                })
            
            for edge in G.edges():
                edges.append({
                    "source": edge[0],
                    "target": edge[1],
                    "weight": G[edge[0]][edge[1]].get('weight', 1)
                })
            
            return {
                "nodes": nodes[:50],  # Limit for visualization
                "edges": edges[:100],
                "total_entities": len(entities),
                "connected_entities": len(G.nodes()),
                "relationships": len(G.edges())
            }
        except Exception as e:
            return {"error": f"Error generating knowledge graph: {str(e)}"}
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type using AI"""
        try:
            classification_prompt = f"""
            Analyze the following document excerpt and classify its type. Consider the writing style, structure, content, and purpose.
            
            Document excerpt (first 2000 characters):
            {text[:2000]}
            
            Classify this document into one of these categories:
            1. Research Paper/Academic Article
            2. Technical Manual/Documentation
            3. Business Report/Proposal
            4. Legal Document/Contract
            5. Educational Material/Textbook
            6. News Article/Journalism
            7. Personal Document/Letter
            8. Government/Policy Document
            9. Financial Report/Analysis
            10. Other
            
            Provide your analysis in this format:
            Document Type: [Primary classification]
            Confidence: [High/Medium/Low]
            Reasoning: [Brief explanation of why you classified it this way]
            Secondary Type: [If applicable]
            Key Indicators: [List 3-5 specific features that led to this classification]
            """
            
            response = self.llm.invoke(classification_prompt)
            classification_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            lines = classification_text.split('\n')
            result = {
                "document_type": "Unknown",
                "confidence": "Low",
                "reasoning": "",
                "secondary_type": "",
                "key_indicators": []
            }
            
            key = None
            indicators_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("Document Type:"):
                    result["document_type"] = line.split(":", 1)[1].strip()
                    key = None
                elif line.startswith("Confidence:"):
                    result["confidence"] = line.split(":", 1)[1].strip()
                    key = None
                elif line.startswith("Reasoning:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                    key = None
                elif line.startswith("Secondary Type:"):
                    result["secondary_type"] = line.split(":", 1)[1].strip()
                    key = None
                elif line.startswith("Key Indicators:"):
                    key = "key_indicators"
                    indicators_text = line.split(":", 1)[1].strip()
                    if indicators_text:
                        indicators_lines.append(indicators_text)
                elif key == "key_indicators":
                    if line and not any(line.startswith(lbl) for lbl in ["Document Type:", "Confidence:", "Reasoning:", "Secondary Type:", "Key Indicators:"]):
                        indicators_lines.append(line)
                    else:
                        key = None
            if indicators_lines:
                # Assume each indicator is on a new line or comma-separated
                indicators = []
                for l in indicators_lines:
                    indicators.extend([ind.strip() for ind in l.split(',') if ind.strip()])
                result["key_indicators"] = indicators
            
            return result
        except Exception as e:
            return {"error": f"Error classifying document: {str(e)}"}
    
    def generate_executive_summary(self, text: str, summary_type: str = "executive") -> str:
        """Generate different types of summaries"""
        try:
            if summary_type == "executive":
                prompt = f"""
                Create a concise executive summary of the following document. Focus on:
                - Key findings and conclusions
                - Main recommendations
                - Business impact
                - Critical decisions needed
                
                Keep it under 200 words and suitable for C-level executives.
                
                Document: {text[:4000]}
                
                Executive Summary:
                """
            elif summary_type == "technical":
                prompt = f"""
                Create a technical summary focusing on:
                - Methodology and approach
                - Technical specifications
                - Implementation details
                - Technical challenges and solutions
                
                Document: {text[:4000]}
                
                Technical Summary:
                """
            else:  # detailed
                prompt = f"""
                Create a comprehensive detailed summary covering:
                - All major sections and topics
                - Key points and supporting details
                - Context and background
                - Implications and next steps
                
                Document: {text[:4000]}
                
                Detailed Summary:
                """
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating summary: {str(e)}"

class DocumentComparison:
    """Compare multiple documents"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def compare_documents(self, doc1_text: str, doc2_text: str, doc1_name: str, doc2_name: str) -> Dict[str, Any]:
        """Compare two documents and find similarities/differences"""
        try:
            comparison_prompt = f"""
            Compare these two documents and provide a detailed analysis:
            
            Document 1 ({doc1_name}):
            {doc1_text[:2000]}
            
            Document 2 ({doc2_name}):
            {doc2_text[:2000]}
            
            Provide comparison in this format:
            
            SIMILARITIES:
            - [List key similarities]
            
            DIFFERENCES:
            - [List key differences]
            
            CONTENT OVERLAP:
            - [Percentage and description of overlapping content]
            
            UNIQUE TO DOCUMENT 1:
            - [Content unique to first document]
            
            UNIQUE TO DOCUMENT 2:
            - [Content unique to second document]
            
            RECOMMENDATION:
            - [Which document to use for what purpose]
            """
            
            response = self.llm.invoke(comparison_prompt)
            comparison_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "comparison_analysis": comparison_text,
                "doc1_name": doc1_name,
                "doc2_name": doc2_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Error comparing documents: {str(e)}"}

class SmartSearch:
    """Advanced search capabilities"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def semantic_search(self, query: str, search_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform semantic search with different strategies"""
        try:
            if search_type == "comprehensive":
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 10, "lambda_mult": 0.5}
                )
            elif search_type == "precise":
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
            else:  # exploratory
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 15, "lambda_mult": 0.3}
                )
            
            # Use similarity_search_with_score if available
            if hasattr(retriever, "similarity_search_with_score"):
                docs_with_scores = retriever.similarity_search_with_score(query)
                results = []
                for i, (doc, score) in enumerate(docs_with_scores):
                    results.append({
                        "rank": i + 1,
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": score
                    })
            else:
                docs = retriever.get_relevant_documents(query)
                results = []
                for i, doc in enumerate(docs):
                    results.append({
                        "rank": i + 1,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
            
            return {
                "query": query,
                "search_type": search_type,
                "total_results": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": f"Error in semantic search: {str(e)}"}
    
    def find_citations_and_references(self, text: str) -> Dict[str, List[str]]:
        """Extract citations and references from text"""
        try:
            # Pattern for academic citations
            citation_patterns = [
                r'\([A-Za-z]+\s+et\s+al\.,?\s+\d{4}\)',  # (Author et al., 2023)
                r'\([A-Za-z]+\s+&\s+[A-Za-z]+,?\s+\d{4}\)',  # (Author & Author, 2023)
                r'\([A-Za-z]+,?\s+\d{4}\)',  # (Author, 2023)
                r'\[\d+\]',  # [1], [2], etc.
            ]
            
            citations = []
            for pattern in citation_patterns:
                citations.extend(re.findall(pattern, text))
            
            # Pattern for URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            
            # Pattern for DOIs
            doi_pattern = r'10\.\d+/[^\s]+'
            dois = re.findall(doi_pattern, text)
            
            return {
                "citations": list(set(citations)),
                "urls": list(set(urls)),
                "dois": list(set(dois)),
                "total_references": len(set(citations + urls + dois))
            }
        except Exception as e:
            return {"error": f"Error extracting references: {str(e)}"}

# Export classes for use in main application
__all__ = ['AdvancedDocumentAnalyzer', 'DocumentComparison', 'SmartSearch']