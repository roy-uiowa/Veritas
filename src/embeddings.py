from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

class WatsonxEmbedding:
    """Wrapper for Watsonx embedding model"""
    
    def __init__(self, project_id="skills-network"):
        self.project_id = project_id
        self.embedding_model = None
        self._initialize_embedding()
    
    def _initialize_embedding(self):
        """Initialize the Watsonx embedding model"""
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }
        
        self.embedding_model = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=self.project_id,
            params=embed_params,
        )
    
    def get_embedding_model(self):
        """Get the embedding model instance"""
        return self.embedding_model
    
    def embed_documents(self, texts):
        """Embed a list of documents"""
        return self.embedding_model.embed_documents(texts)
    
    def embed_query(self, text):
        """Embed a single query"""
        return self.embedding_model.embed_query(text)
