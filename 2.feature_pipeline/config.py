from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):

    # Embeddings config
    _file_path: str = '1.datacollection_pipeline/datasets/'
    ground_truth_csv: str = '1.datacollection_pipeline/datasets/ground_truth.csv'

    OPENAI_MODEL_ID: str = "gpt-4"
    OPENAI_API_KEY: str = ""
    
    MONGO_DB_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "product-core"
    MONGO_COLLECTION_NAME: str = "xngen-knowledge-base"
    # Embeddings config
    EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 256
    EMBEDDING_SIZE: int = 384
    
    # Memgraph config
    MEMGRAPH_URI = "bolt://localhost:7687"
    

settings = AppSettings()
