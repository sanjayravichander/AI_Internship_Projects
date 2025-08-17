"""
Advanced Features for Document Intelligence Chatbot
==================================================

This module provides advanced document analysis, comparison, and search capabilities
for the Document Intelligence Chatbot application.

Author: AI Intern
Version: 1.0.0
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import hashlib
from datetime import datetime

class AdvancedDocumentAnalyzer:
    """Advanced document analysis capabilities"""
    
    def __init__(self, vectorstore=None, llm=None):
        self.analysis_results = {}
        self.supported_formats = ['.pdf', '.txt', '.docx']
        self.vectorstore = vectorstore
        self.llm = llm
        
    def analyze_document(self, document_text: str, document_name: str = "document") -> Dict[str, Any]:
        """
        Analyze document content and extract comprehensive insights
        
        Args:
            document_text (str): The text content of the document
            document_name (str): Name of the document for reference
            
        Returns:
            Dict containing analysis results
        """
        try:
            analysis = {
                "document_name": document_name,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(document_text.split()),
                "character_count": len(document_text),
                "paragraph_count": len([p for p in document_text.split('\n\n') if p.strip()]),
                "sentence_count": len([s for s in re.split(r'[.!?]+', document_text) if s.strip()]),
                "readability_score": self._calculate_readability(document_text),
                "key_topics": self._extract_key_topics(document_text),
                "sentiment_analysis": self._analyze_sentiment(document_text),
                "language_complexity": self._analyze_complexity(document_text),
                "document_structure": self._analyze_structure(document_text)
            }
            
            # Store results for future reference
            doc_hash = hashlib.md5(document_text.encode()).hexdigest()
            self.analysis_results[doc_hash] = analysis
            
            return {
                "status": "success",
                "insights": analysis,
                "summary": self._generate_summary(analysis)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "insights": {}
            }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (lower is easier)
        readability = (avg_sentence_length * 0.5) + (avg_word_length * 2.0)
        return round(readability, 2)
    
    def _extract_key_topics(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key topics from the document"""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
            'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
            'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
            'such', 'take', 'than', 'them', 'well', 'were', 'what'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = Counter(filtered_words)
        
        return [word for word, freq in word_freq.most_common(top_n)]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'positive', 'success', 'achievement', 'benefit', 'advantage',
            'improvement', 'effective', 'efficient', 'valuable', 'important'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'problem',
            'issue', 'error', 'failure', 'disadvantage', 'difficulty',
            'challenge', 'concern', 'risk', 'threat', 'danger'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            confidence = negative_ratio
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 2),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze language complexity"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if not words or not sentences:
            return {"complexity": "low", "score": 0}
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        # Count complex words (more than 6 characters)
        complex_words = [word for word in words if len(word) > 6]
        complex_word_ratio = len(complex_words) / len(words)
        
        # Calculate complexity score
        complexity_score = (avg_word_length * 0.3) + (avg_sentence_length * 0.4) + (complex_word_ratio * 0.3)
        
        if complexity_score < 3:
            complexity_level = "low"
        elif complexity_score < 6:
            complexity_level = "medium"
        else:
            complexity_level = "high"
        
        return {
            "complexity": complexity_level,
            "score": round(complexity_score, 2),
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "complex_word_ratio": round(complex_word_ratio * 100, 2)
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = text.split('\n')
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        # Look for headers (lines that are short and might be titles)
        potential_headers = []
        for line in lines:
            line = line.strip()
            if line and len(line) < 100 and not line.endswith('.'):
                # Check if it might be a header
                if line.isupper() or line.istitle():
                    potential_headers.append(line)
        
        return {
            "total_lines": len(lines),
            "total_paragraphs": len(paragraphs),
            "potential_headers": len(potential_headers),
            "avg_paragraph_length": round(sum(len(p.split()) for p in paragraphs) / len(paragraphs), 2) if paragraphs else 0,
            "structure_type": "structured" if potential_headers else "unstructured"
        }
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the analysis"""
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"Document contains {analysis['word_count']} words in {analysis['paragraph_count']} paragraphs.")
        
        # Complexity
        complexity = analysis['language_complexity']['complexity']
        summary_parts.append(f"Language complexity is {complexity}.")
        
        # Sentiment
        sentiment_info = analysis['sentiment_analysis']
        summary_parts.append(f"Overall sentiment appears {sentiment_info['sentiment']} with {sentiment_info['confidence']}% confidence.")
        
        # Key topics
        if analysis['key_topics']:
            top_topics = ', '.join(analysis['key_topics'][:5])
            summary_parts.append(f"Key topics include: {top_topics}.")
        
        return ' '.join(summary_parts)
    
    def analyze_document_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze document complexity and readability"""
        try:
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            
            if not words or not sentences:
                return {"error": "Insufficient text for analysis"}
            
            # Calculate readability metrics
            avg_word_length = sum(len(word) for word in words) / len(words)
            avg_sentence_length = len(words) / len(sentences)
            
            # Flesch Reading Ease Score (simplified)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
            flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
            
            # Flesch-Kincaid Grade Level (simplified)
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * (avg_word_length / 4.7)) - 15.59
            fk_grade = max(1, fk_grade)  # Minimum grade 1
            
            # Determine complexity level
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
            
            # Calculate vocabulary richness
            unique_words = len(set(word.lower() for word in words))
            vocabulary_richness = unique_words / len(words) if words else 0
            
            # Estimate reading time (average 200 words per minute)
            estimated_reading_time = len(words) / 200
            
            return {
                "complexity_level": complexity_level,
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_kincaid_grade": round(fk_grade, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2),
                "vocabulary_richness": round(vocabulary_richness, 3),
                "total_words": len(words),
                "total_sentences": len(sentences),
                "estimated_reading_time": round(estimated_reading_time, 1)
            }
        except Exception as e:
            return {"error": f"Error analyzing complexity: {str(e)}"}
    
    def extract_key_entities(self, text: str) -> Dict[str, Any]:
        """Extract key entities from text using simple pattern matching"""
        try:
            # Simple entity extraction using patterns
            entities = {
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "numbers": [],
                "emails": [],
                "urls": []
            }
            
            # Extract emails
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            entities["emails"] = list(set(re.findall(email_pattern, text)))
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            entities["urls"] = list(set(re.findall(url_pattern, text)))
            
            # Extract dates (simple patterns)
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ]
            for pattern in date_patterns:
                entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
            entities["dates"] = list(set(entities["dates"]))
            
            # Extract numbers (including percentages, currency)
            number_pattern = r'\b\d+(?:\.\d+)?%?\b|\$\d+(?:,\d{3})*(?:\.\d{2})?\b'
            entities["numbers"] = list(set(re.findall(number_pattern, text)))
            
            # Extract potential names (capitalized words, but filter common words)
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            common_words = {
                'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'So', 'Yet',
                'For', 'Nor', 'As', 'At', 'By', 'In', 'Of', 'On', 'To', 'Up', 'With',
                'From', 'Into', 'Upon', 'About', 'Above', 'Across', 'After', 'Against',
                'Along', 'Among', 'Around', 'Before', 'Behind', 'Below', 'Beneath',
                'Beside', 'Between', 'Beyond', 'During', 'Except', 'Inside', 'Outside',
                'Through', 'Throughout', 'Under', 'Until', 'Within', 'Without'
            }
            potential_names = [word for word in words if word not in common_words]
            
            # Simple heuristic: if a capitalized word appears multiple times, it might be important
            name_counts = Counter(potential_names)
            entities["people"] = [name for name, count in name_counts.most_common(20) if count > 1]
            
            # Extract potential organizations (words ending with common org suffixes)
            org_patterns = [
                r'\b[A-Z][a-zA-Z\s]+(?:Inc|Corp|LLC|Ltd|Company|Corporation|Institute|University|College|School|Department|Agency|Bureau|Office|Center|Centre|Foundation|Association|Society|Organization|Group|Team|Committee|Board|Council|Commission)\b',
                r'\b(?:Inc|Corp|LLC|Ltd|Company|Corporation|Institute|University|College|School|Department|Agency|Bureau|Office|Center|Centre|Foundation|Association|Society|Organization|Group|Team|Committee|Board|Council|Commission)\b'
            ]
            for pattern in org_patterns:
                entities["organizations"].extend(re.findall(pattern, text, re.IGNORECASE))
            entities["organizations"] = list(set(entities["organizations"]))
            
            # Extract potential locations (capitalized words that might be places)
            # This is very basic - in a real implementation, you'd use a gazetteer or NER model
            location_indicators = ['City', 'State', 'Country', 'County', 'Province', 'Region', 'District', 'Area']
            for indicator in location_indicators:
                pattern = r'\b[A-Z][a-zA-Z\s]+' + indicator + r'\b'
                entities["locations"].extend(re.findall(pattern, text))
            entities["locations"] = list(set(entities["locations"]))
            
            # Calculate total entities
            total_entities = sum(len(entity_list) for entity_list in entities.values())
            
            return {
                "entities": entities,
                "total_entities": total_entities,
                "entity_summary": {
                    "people_count": len(entities["people"]),
                    "organizations_count": len(entities["organizations"]),
                    "locations_count": len(entities["locations"]),
                    "dates_count": len(entities["dates"]),
                    "numbers_count": len(entities["numbers"]),
                    "emails_count": len(entities["emails"]),
                    "urls_count": len(entities["urls"])
                }
            }
        except Exception as e:
            return {"error": f"Error extracting entities: {str(e)}"}
    
    def classify_document_type(self, text: str) -> Dict[str, Any]:
        """Classify document type based on content patterns"""
        try:
            text_lower = text.lower()
            
            # Define document type indicators
            document_types = {
                "Research Paper": [
                    "abstract", "introduction", "methodology", "results", "conclusion",
                    "references", "bibliography", "hypothesis", "experiment", "analysis"
                ],
                "Technical Manual": [
                    "installation", "configuration", "setup", "troubleshooting",
                    "user guide", "manual", "instructions", "procedure", "step"
                ],
                "Business Report": [
                    "executive summary", "quarterly", "annual", "revenue", "profit",
                    "market", "business", "financial", "performance", "strategy"
                ],
                "Legal Document": [
                    "whereas", "therefore", "party", "agreement", "contract",
                    "terms", "conditions", "legal", "law", "court", "jurisdiction"
                ],
                "Academic Paper": [
                    "university", "college", "professor", "student", "course",
                    "curriculum", "education", "learning", "academic", "study"
                ],
                "News Article": [
                    "reported", "according to", "sources", "journalist", "news",
                    "breaking", "update", "story", "coverage", "press"
                ],
                "Policy Document": [
                    "policy", "regulation", "compliance", "standard", "guideline",
                    "requirement", "mandatory", "shall", "must", "prohibited"
                ]
            }
            
            # Score each document type
            type_scores = {}
            for doc_type, indicators in document_types.items():
                score = sum(1 for indicator in indicators if indicator in text_lower)
                type_scores[doc_type] = score
            
            # Find the best match
            if type_scores:
                best_type = max(type_scores, key=type_scores.get)
                best_score = type_scores[best_type]
                
                # Calculate confidence based on score
                max_possible_score = len(document_types[best_type])
                confidence = (best_score / max_possible_score) * 100 if max_possible_score > 0 else 0
                
                # Determine confidence level
                if confidence >= 50:
                    confidence_level = "High"
                elif confidence >= 25:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                    best_type = "General Document"  # Default for low confidence
                
                return {
                    "document_type": best_type,
                    "confidence": confidence_level,
                    "confidence_score": round(confidence, 2),
                    "type_scores": type_scores,
                    "reasoning": f"Identified as {best_type} based on {best_score} matching indicators"
                }
            else:
                return {
                    "document_type": "General Document",
                    "confidence": "Low",
                    "confidence_score": 0,
                    "reasoning": "Could not identify specific document type"
                }
        except Exception as e:
            return {"error": f"Error classifying document: {str(e)}"}
    
    def generate_executive_summary(self, text: str, summary_type: str = "executive") -> str:
        """Generate different types of summaries"""
        try:
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if not sentences:
                return "Insufficient content for summary generation."
            
            # Score sentences based on word frequency
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = Counter(words)
            
            # Remove common stop words
            stop_words = {
                'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
                'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
                'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over',
                'such', 'take', 'than', 'them', 'well', 'were', 'what', 'said',
                'each', 'which', 'their', 'would', 'there', 'could', 'other'
            }
            
            # Score sentences
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                sentence_words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
                score = sum(word_freq.get(word, 0) for word in sentence_words if word not in stop_words)
                sentence_scores[i] = score
            
            # Select top sentences based on summary type
            if summary_type == "executive":
                num_sentences = min(5, len(sentences) // 4)  # 25% of sentences, max 5
            elif summary_type == "technical":
                num_sentences = min(8, len(sentences) // 3)  # 33% of sentences, max 8
            else:  # detailed
                num_sentences = min(12, len(sentences) // 2)  # 50% of sentences, max 12
            
            # Get top scoring sentences
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
            
            # Sort by original order
            top_sentences = sorted(top_sentences, key=lambda x: x[0])
            
            # Create summary
            summary_sentences = [sentences[i] for i, _ in top_sentences]
            summary = '. '.join(summary_sentences) + '.'
            
            # Add summary type prefix
            if summary_type == "executive":
                prefix = "Executive Summary: "
            elif summary_type == "technical":
                prefix = "Technical Summary: "
            else:
                prefix = "Detailed Summary: "
            
            return prefix + summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_knowledge_graph(self, text: str) -> Dict[str, Any]:
        """Generate a simple knowledge graph from text"""
        try:
            # Extract entities
            entities_result = self.extract_key_entities(text)
            if "error" in entities_result:
                return entities_result
            
            entities = entities_result["entities"]
            
            # Create nodes for the graph
            nodes = []
            node_id = 0
            entity_to_id = {}
            
            # Add entity nodes
            for entity_type, entity_list in entities.items():
                for entity in entity_list[:10]:  # Limit to top 10 per type
                    nodes.append({
                        "id": entity,
                        "label": entity,
                        "type": entity_type.upper(),
                        "size": 10
                    })
                    entity_to_id[entity] = node_id
                    node_id += 1
            
            # Create simple relationships based on co-occurrence
            edges = []
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences[:50]:  # Limit to first 50 sentences for performance
                sentence_entities = []
                for entity_type, entity_list in entities.items():
                    for entity in entity_list:
                        if entity.lower() in sentence.lower():
                            sentence_entities.append(entity)
                
                # Create edges between entities that appear in the same sentence
                for i, entity1 in enumerate(sentence_entities):
                    for entity2 in sentence_entities[i+1:]:
                        if entity1 != entity2:
                            edges.append({
                                "source": entity1,
                                "target": entity2,
                                "weight": 1,
                                "relationship": "co-occurs"
                            })
            
            # Remove duplicate edges and count relationships
            edge_counts = {}
            for edge in edges:
                key = tuple(sorted([edge["source"], edge["target"]]))
                edge_counts[key] = edge_counts.get(key, 0) + 1
            
            # Create final edges with weights
            final_edges = []
            for (source, target), weight in edge_counts.items():
                if weight > 1:  # Only include relationships that occur multiple times
                    final_edges.append({
                        "source": source,
                        "target": target,
                        "weight": weight,
                        "relationship": "co-occurs"
                    })
            
            return {
                "nodes": nodes,
                "edges": final_edges,
                "total_entities": len(nodes),
                "connected_entities": len([node for node in nodes if any(edge["source"] == node["id"] or edge["target"] == node["id"] for edge in final_edges)]),
                "relationships": len(final_edges),
                "graph_density": len(final_edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
            }
            
        except Exception as e:
            return {"error": f"Error generating knowledge graph: {str(e)}"}


class DocumentComparison:
    """Compare multiple documents for similarities and differences"""
    
    def __init__(self, llm=None):
        self.comparison_cache = {}
        self.llm = llm
    
    def compare_documents(self, doc1_text: str, doc2_text: str, doc1_name: str = "Document 1", doc2_name: str = "Document 2") -> Dict[str, Any]:
        """
        Compare two documents and identify similarities/differences
        
        Args:
            doc1_text (str): Text content of first document
            doc2_text (str): Text content of second document
            doc1_name (str): Name of first document
            doc2_name (str): Name of second document
            
        Returns:
            Dict containing comparison results
        """
        try:
            # Calculate similarity metrics
            similarity_score = self._calculate_similarity(doc1_text, doc2_text)
            word_overlap = self._calculate_word_overlap(doc1_text, doc2_text)
            structural_comparison = self._compare_structure(doc1_text, doc2_text)
            
            # Find common themes
            common_topics = self._find_common_topics(doc1_text, doc2_text)
            
            # Calculate differences
            differences = self._identify_differences(doc1_text, doc2_text)
            
            comparison_result = {
                "document_1": doc1_name,
                "document_2": doc2_name,
                "similarity_score": similarity_score,
                "word_overlap": word_overlap,
                "structural_comparison": structural_comparison,
                "common_topics": common_topics,
                "differences": differences,
                "recommendation": self._generate_comparison_recommendation(similarity_score, word_overlap)
            }
            
            return {
                "status": "success",
                "comparison": comparison_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "comparison": {}
            }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        words1 = set(re.findall(r'\b[a-zA-Z]+\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        return round(jaccard_similarity * 100, 2)
    
    def _calculate_word_overlap(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate word overlap statistics"""
        words1 = re.findall(r'\b[a-zA-Z]+\b', text1.lower())
        words2 = re.findall(r'\b[a-zA-Z]+\b', text2.lower())
        
        set1 = set(words1)
        set2 = set(words2)
        
        common_words = set1.intersection(set2)
        unique_to_doc1 = set1 - set2
        unique_to_doc2 = set2 - set1
        
        return {
            "common_words_count": len(common_words),
            "unique_to_doc1": len(unique_to_doc1),
            "unique_to_doc2": len(unique_to_doc2),
            "common_words": list(common_words)[:20],  # Top 20 common words
            "overlap_percentage": round((len(common_words) / len(set1.union(set2))) * 100, 2) if set1.union(set2) else 0
        }
    
    def _compare_structure(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare structural elements of documents"""
        def get_structure_stats(text):
            return {
                "paragraphs": len([p for p in text.split('\n\n') if p.strip()]),
                "sentences": len(re.split(r'[.!?]+', text)),
                "avg_sentence_length": len(text.split()) / max(1, len(re.split(r'[.!?]+', text)))
            }
        
        struct1 = get_structure_stats(text1)
        struct2 = get_structure_stats(text2)
        
        return {
            "document_1_structure": struct1,
            "document_2_structure": struct2,
            "structure_similarity": self._calculate_structure_similarity(struct1, struct2)
        }
    
    def _calculate_structure_similarity(self, struct1: Dict, struct2: Dict) -> float:
        """Calculate structural similarity between documents"""
        # Simple structural similarity based on paragraph and sentence ratios
        para_ratio = min(struct1["paragraphs"], struct2["paragraphs"]) / max(struct1["paragraphs"], struct2["paragraphs"], 1)
        sent_ratio = min(struct1["sentences"], struct2["sentences"]) / max(struct1["sentences"], struct2["sentences"], 1)
        
        return round((para_ratio + sent_ratio) / 2 * 100, 2)
    
    def _find_common_topics(self, text1: str, text2: str) -> List[str]:
        """Find common topics between documents"""
        # Extract keywords from both documents
        words1 = re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower())
        words2 = re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower())
        
        # Count frequencies
        freq1 = Counter(words1)
        freq2 = Counter(words2)
        
        # Find common words with significant frequency in both
        common_topics = []
        for word in freq1:
            if word in freq2 and freq1[word] > 2 and freq2[word] > 2:
                common_topics.append(word)
        
        return sorted(common_topics, key=lambda x: freq1[x] + freq2[x], reverse=True)[:10]
    
    def _identify_differences(self, text1: str, text2: str) -> Dict[str, Any]:
        """Identify key differences between documents"""
        words1 = Counter(re.findall(r'\b[a-zA-Z]{4,}\b', text1.lower()))
        words2 = Counter(re.findall(r'\b[a-zA-Z]{4,}\b', text2.lower()))
        
        # Find words that appear significantly more in one document
        unique_to_doc1 = []
        unique_to_doc2 = []
        
        for word, count in words1.most_common(20):
            if words2.get(word, 0) < count / 2:  # Word appears much more in doc1
                unique_to_doc1.append(word)
        
        for word, count in words2.most_common(20):
            if words1.get(word, 0) < count / 2:  # Word appears much more in doc2
                unique_to_doc2.append(word)
        
        return {
            "unique_themes_doc1": unique_to_doc1[:10],
            "unique_themes_doc2": unique_to_doc2[:10],
            "length_difference": abs(len(text1.split()) - len(text2.split())),
            "complexity_difference": "Analysis would require individual document analysis"
        }
    
    def _generate_comparison_recommendation(self, similarity_score: float, word_overlap: Dict) -> str:
        """Generate recommendation based on comparison results"""
        if similarity_score > 70:
            return "Documents are highly similar and likely cover the same topics."
        elif similarity_score > 40:
            return "Documents have moderate similarity with some overlapping themes."
        elif similarity_score > 20:
            return "Documents have some common elements but are largely different."
        else:
            return "Documents are quite different and cover distinct topics."


class SmartSearch:
    """Enhanced semantic search capabilities"""
    
    def __init__(self, vectorstore=None, llm=None):
        self.search_history = []
        self.search_cache = {}
        self.vectorstore = vectorstore
        self.llm = llm
    
    def semantic_search(self, query: str, documents: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """
        Perform semantic search across documents
        
        Args:
            query (str): Search query
            documents (List[Dict]): List of documents with 'content' and 'metadata'
            top_k (int): Number of top results to return
            
        Returns:
            Dict containing search results
        """
        try:
            # Simple semantic search implementation
            query_words = set(re.findall(r'\b[a-zA-Z]+\b', query.lower()))
            
            results = []
            for i, doc in enumerate(documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(query_words, content)
                
                if relevance_score > 0:
                    # Find relevant snippets
                    snippets = self._extract_relevant_snippets(query, content)
                    
                    results.append({
                        "document_index": i,
                        "relevance_score": relevance_score,
                        "metadata": metadata,
                        "snippets": snippets,
                        "match_count": len([word for word in query_words if word in content.lower()])
                    })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Store search in history
            self.search_history.append({
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results_count": len(results)
            })
            
            return {
                "status": "success",
                "query": query,
                "total_results": len(results),
                "results": results[:top_k],
                "search_suggestions": self._generate_search_suggestions(query, results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "results": []
            }
    
    def _calculate_relevance(self, query_words: set, content: str) -> float:
        """Calculate relevance score for a document"""
        content_words = set(re.findall(r'\b[a-zA-Z]+\b', content.lower()))
        
        if not content_words:
            return 0.0
        
        # Calculate different relevance factors
        exact_matches = len(query_words.intersection(content_words))
        partial_matches = 0
        
        # Check for partial word matches
        for query_word in query_words:
            for content_word in content_words:
                if query_word in content_word or content_word in query_word:
                    partial_matches += 0.5
        
        # Calculate TF-IDF-like score (simplified)
        total_words = len(content.split())
        term_frequency = sum(content.lower().count(word) for word in query_words) / total_words
        
        # Combine scores
        relevance_score = (exact_matches * 2) + partial_matches + (term_frequency * 10)
        
        return round(relevance_score, 3)
    
    def _extract_relevant_snippets(self, query: str, content: str, snippet_length: int = 200) -> List[str]:
        """Extract relevant snippets from content"""
        query_words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        sentences = re.split(r'[.!?]+', content)
        
        relevant_snippets = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > 0:
                # Create snippet around the sentence
                start_pos = max(0, content.find(sentence) - snippet_length // 2)
                end_pos = min(len(content), start_pos + snippet_length)
                snippet = content[start_pos:end_pos].strip()
                
                if snippet and snippet not in relevant_snippets:
                    relevant_snippets.append(snippet)
        
        return relevant_snippets[:3]  # Return top 3 snippets
    
    def _generate_search_suggestions(self, query: str, results: List[Dict]) -> List[str]:
        """Generate search suggestions based on results"""
        suggestions = []
        
        # Extract common words from top results
        if results:
            all_words = []
            for result in results[:3]:  # Top 3 results
                for snippet in result.get('snippets', []):
                    words = re.findall(r'\b[a-zA-Z]{4,}\b', snippet.lower())
                    all_words.extend(words)
            
            # Find most common words that aren't in the original query
            query_words = set(re.findall(r'\b[a-zA-Z]+\b', query.lower()))
            word_freq = Counter(all_words)
            
            for word, freq in word_freq.most_common(5):
                if word not in query_words and freq > 1:
                    suggestions.append(f"{query} {word}")
        
        return suggestions[:3]
    
    def get_search_history(self) -> List[Dict]:
        """Get search history"""
        return self.search_history[-10:]  # Return last 10 searches
    
    def clear_search_history(self):
        """Clear search history"""
        self.search_history = []
        self.search_cache = {}
    
    def semantic_search(self, query: str, search_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform semantic search using the vectorstore"""
        try:
            if not self.vectorstore:
                return {"error": "Vectorstore not available for search"}
            
            # Adjust search parameters based on search type
            if search_type == "precise":
                search_kwargs = {"k": 3}
            elif search_type == "exploratory":
                search_kwargs = {"k": 10, "search_type": "mmr", "lambda_mult": 0.7}
            else:  # comprehensive
                search_kwargs = {"k": 6}
            
            # Perform search
            retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
            docs = retriever.get_relevant_documents(query)
            
            # Format results
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "rank": i + 1,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1.0 - (i * 0.1)  # Simple scoring
                })
            
            return {
                "status": "success",
                "query": query,
                "search_type": search_type,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}
    
    def find_citations_and_references(self, text: str) -> Dict[str, Any]:
        """Find citations and references in text"""
        try:
            citations = []
            urls = []
            dois = []
            
            # Find DOIs
            doi_pattern = r'10\.\d{4,}\/[^\s]+'
            dois = list(set(re.findall(doi_pattern, text)))
            
            # Find URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = list(set(re.findall(url_pattern, text)))
            
            # Find citation patterns
            citation_patterns = [
                r'\([A-Za-z]+,?\s+\d{4}\)',  # (Author, Year)
                r'\[[0-9,\s-]+\]',           # [1], [1-3], [1,2,3]
                r'[A-Za-z]+\s+et\s+al\.\s+\(\d{4}\)',  # Author et al. (Year)
            ]
            
            for pattern in citation_patterns:
                citations.extend(re.findall(pattern, text))
            
            citations = list(set(citations))
            
            return {
                "citations": citations,
                "urls": urls,
                "dois": dois,
                "total_references": len(citations) + len(urls) + len(dois)
            }
            
        except Exception as e:
            return {"error": f"Error finding references: {str(e)}"}