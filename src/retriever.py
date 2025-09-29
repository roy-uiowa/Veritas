from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List, Dict, Any, Optional
from .embeddings import WatsonxEmbedding

class PolicyRetriever:
    """Policy document retriever with multiple search methods"""
    
    def __init__(self, embedding_model: WatsonxEmbedding):
        self.embedding_model = embedding_model
        self.vector_db = None
        self.retriever = None
    
    def load_documents(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 20):
        """Load and chunk documents from file"""
        # Load documents
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_db = Chroma.from_documents(
            chunks, 
            self.embedding_model.get_embedding_model()
        )
        
        return chunks
    
    def setup_retriever(self, search_type: str = "similarity", **kwargs):
        """Setup retriever with specified search type and parameters"""
        if self.vector_db is None:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        search_kwargs = {}
        
        if search_type == "similarity":
            if "k" in kwargs:
                search_kwargs["k"] = kwargs["k"]
            if "score_threshold" in kwargs:
                search_type = "similarity_score_threshold"
                search_kwargs["score_threshold"] = kwargs["score_threshold"]
        
        elif search_type == "mmr":
            if "k" in kwargs:
                search_kwargs["k"] = kwargs["k"]
            if "fetch_k" in kwargs:
                search_kwargs["fetch_k"] = kwargs["fetch_k"]
            if "lambda_mult" in kwargs:
                search_kwargs["lambda_mult"] = kwargs["lambda_mult"]
        
        self.retriever = self.vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return self.retriever
    
    def search(self, query: str, search_type: str = "similarity", **kwargs) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        if self.retriever is None or kwargs:
            self.setup_retriever(search_type, **kwargs)
        
        if self.retriever is None:
            raise ValueError("Retriever not initialized")
        
        docs = self.retriever.invoke(query)
        
        # Format results
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None)  # Some retrievers include scores
            })
        
        return results
    
    def similarity_search(self, query: str, k: int = 4, score_threshold: Optional[float] = None):
        """Perform similarity search"""
        kwargs = {"k": k}
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        
        return self.search(query, "similarity", **kwargs)
    
    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5):
        """Perform MMR search"""
        return self.search(
            query, 
            "mmr", 
            k=k, 
            fetch_k=fetch_k, 
            lambda_mult=lambda_mult
        )
    
    def get_available_search_types(self) -> List[str]:
        """Get list of available search types"""
        return ["similarity", "mmr"]
