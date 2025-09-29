from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
from typing import List, Dict, Any
from .retriever import PolicyRetriever
from .embeddings import WatsonxEmbedding

class PolicyAnalyzer:
    """Main policy analyzer class that combines retrieval and generation"""
    
    def __init__(self, project_id: str = "skills-network"):
        self.project_id = project_id
        self.llm = None
        self.embedding_model = None
        self.retriever = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM and embedding models"""
        # Initialize LLM
        model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'
        
        parameters = {
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.TEMPERATURE: 0.5,
        }
        
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com"
        }
        
        model = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=credentials,
            project_id=self.project_id
        )
        
        self.llm = WatsonxLLM(model=model)
        
        # Initialize embedding model
        self.embedding_model = WatsonxEmbedding(self.project_id)
        
        # Initialize retriever
        self.retriever = PolicyRetriever(self.embedding_model)
    
    def load_policies(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 20):
        """Load policy documents"""
        return self.retriever.load_documents(file_path, chunk_size, chunk_overlap)
    
    def ask(self, 
            query: str, 
            search_type: str = "similarity", 
            k: int = 4,
            score_threshold: Optional[float] = None,
            include_sources: bool = True) -> Dict[str, Any]:
        """Ask a question about policies"""
        
        # Retrieve relevant documents
        if search_type == "similarity":
            retrieved_docs = self.retriever.similarity_search(
                query, k=k, score_threshold=score_threshold
            )
        elif search_type == "mmr":
            retrieved_docs = self.retriever.mmr_search(query, k=k)
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        if not retrieved_docs:
            return {
                "answer": "No relevant policy information found for your query.",
                "sources": []
            }
        
        # Build context from retrieved documents
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        
        # Create prompt
        prompt = f"""Based on the following company policies, please answer the user's question.

Company Policies:
{context}

Question: {query}

Please provide a clear, concise answer based only on the policies above. If the information isn't available in the policies, state that clearly.

Answer:"""
        
        # Generate answer
        answer = self.llm.invoke(prompt)
        
        result = {
            "answer": answer.strip(),
            "search_type": search_type,
            "parameters": {
                "k": k,
                "score_threshold": score_threshold
            }
        }
        
        if include_sources:
            result["sources"] = retrieved_docs
        
        return result
    
    def get_available_search_types(self) -> List[str]:
        """Get available search types"""
        return self.retriever.get_available_search_types()
    
    def test_retrieval(self, query: str, search_type: str = "similarity", **kwargs):
        """Test retrieval without generation"""
        return self.retriever.search(query, search_type, **kwargs)
