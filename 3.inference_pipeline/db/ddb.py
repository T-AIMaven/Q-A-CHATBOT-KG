from llama_index.core import KnowledgeGraphIndex, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import load_index_from_storage

from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from bson import json_util
from dotenv import load_dotenv
from typing import List
import pandas as pd
import logging
import time
import json
import math
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()
# At the top of your file, after imports
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

# MongoDB setup
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
client = MongoClient(MONGO_DB_URL)
db = client['product-core']
index_collection = db['xngen-knowledge-base']

def split_metadata(metadata, max_length=1000):
    if len(json.dumps(metadata)) <= max_length:
        return metadata
    
    split_metadata = {}
    for key, value in metadata.items():
        if len(json.dumps({key: value})) > max_length:
            split_metadata[key] = str(value)[:max_length]
        else:
            split_metadata[key] = value
    return split_metadata

class StructuredSearchConnector:
    def __init__(self):
        self.documents = self.load_documents()
        self.llm = self.setup_llm()
        self.embedding = embedding_model
        self.index = self.create_or_load_index("nereus")

    def load_documents(self) -> List[Document]:
        start_time = time.time()
        # Read the CSV file
        df = pd.read_csv("2.csv")
        documents = []
        for _, row in df.iterrows():
            json_doc = row.to_dict()
            doc_id = str(len(documents))
            text = "\n".join([f"{key.capitalize()}: {value}" for key, value in json_doc.items() if pd.notna(value)])
            metadata = {key: value for key, value in json_doc.items() if pd.notna(value)}
            doc = Document(
                id=doc_id,
                text=text
                # metadata=metadata
            )
            documents.append(doc)
        end_time = time.time()
        print(f"Load documents time: {end_time - start_time} seconds")
        
        return documents

    def setup_llm(self) -> LangChainLLM:
        api_key = os.getenv("OPENAI_API_KEY")
        return LangChainLLM(llm=ChatOpenAI(model="gpt-4", api_key=api_key))

    def setup_embedding(self) -> HuggingFaceEmbedding:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model_name=model_name)

    def create_or_load_index(self, customer_id: str) -> KnowledgeGraphIndex:
        index_chunks = list(index_collection.find({"customer_id": customer_id}).sort("chunk_index", 1))
        if index_chunks:
            start_time = time.time()
            total_chunks = index_chunks[0]["total_chunks"]
            index_json = "".join(chunk["data"] for chunk in index_chunks)
            
            try:
                serialized_context = json.loads(index_json)
                storage_context = StorageContext.from_dict(serialized_context)
                
                index = load_index_from_storage(storage_context, embed_model=self.embedding)
                logger.info(f"Loaded index from MongoDB ({total_chunks} chunks).")
                end_time = time.time()
                logger.info(f"Load documents existed time: {end_time - start_time} seconds")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
                logger.info("Creating new index due to JSON decode error.")
                index = self.create_new_index(customer_id)
        else:
            index = self.create_new_index(customer_id)
        return index

    def create_new_index(self, customer_id: str) -> KnowledgeGraphIndex:
        start_time = time.time()
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        text_splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=20)
        index = KnowledgeGraphIndex.from_documents(
            self.documents,
            max_triplets_per_chunk=20,
            include_embeddings=True,
            embed_model=self.embedding,
            node_parser=text_splitter,
            storage_context=storage_context,
        )
        self.save_index_to_mongodb(index, customer_id)
        end_time = time.time()
        logger.info(f"Create new index time: {end_time - start_time} seconds")
        return index

    def save_index_to_mongodb(self, index: KnowledgeGraphIndex, customer_id: str):
        start_time = time.time()
        storage_context = index.storage_context
        serialized_context = storage_context.to_dict()
        index_json = json.dumps(serialized_context, default=json_util.default)
        
        chunk_size = 15 * 1024 * 1024  # 15MB chunks
        total_chunks = math.ceil(len(index_json) / chunk_size)
        
        index_collection.delete_many({"customer_id": customer_id})
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = index_json[start:end]
            
            index_collection.insert_one({
                "customer_id": customer_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "data": chunk
            })
        
        logger.info(f"Saved index to MongoDB in {total_chunks} chunks.")
        end_time = time.time()
        logger.info(f"Save index time: {end_time - start_time} seconds")
    def structured_search_index(self, query: str) -> dict:
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            include_text=True,
            response_mode="tree_summarize",
            embedding_mode="hybrid",
            similarity_top_k=5,  # Adjust this to control the number of relevant nodes
            max_tokens=4096  # Adjust based on your model's capabilities
        )

        response = query_engine.query(query)

        print("Response: ", response.response)
        return {"type": "structured", "response": response.response}

def structured_search_query(query: str) -> dict:
    connector = StructuredSearchConnector()
    return connector.structured_search_index(query=query)

if __name__ == "__main__":
    start_time = time.time()
    # Example usage
    query = "who is the CEO of Xngen AI?"
    response = structured_search_query(query=query)
    print(response)
    end_time = time.time()
    logger.info(f"total time: {end_time - start_time} seconds")