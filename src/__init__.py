"""
Veritas - Company Policy Analyzer
AI-powered policy analysis chatbot using vector databases and retrieval techniques.
"""

__version__ = "1.0.0"
__author__ = "Veritas Team: Tarun Roy"

from .embeddings import WatsonxEmbedding
from .retriever import PolicyRetriever
from .analyzer import PolicyAnalyzer

__all__ = ["WatsonxEmbedding", "PolicyRetriever", "PolicyAnalyzer"]
