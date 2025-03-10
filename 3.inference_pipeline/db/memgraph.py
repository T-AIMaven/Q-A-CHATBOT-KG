from neo4j import GraphDatabase
import openai
from typing import List, Dict
from config import settings

openai.api_key = settings.OPENAI_API_KEY

MEMGRAPH_URI = settings.MEMGRAPH_URI
AUTH = ("", "")

class MemgraphSearchConnector:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def query_graph(self, user_query: str) -> List[Dict]:
        """
        Query the graph database and return results.
        """
        with self.driver.session() as session:
            # Use a more general Cypher query to match nodes and relationships
            cypher_query = f"""
            MATCH (n)-[r]->(m)
            WHERE n.value CONTAINS $query OR m.value CONTAINS $query
            RETURN labels(n)[0] AS source_type, n.value AS source, 
                   type(r) AS relationship, 
                   labels(m)[0] AS target_type, m.value AS target
            LIMIT 10
            """
            results = session.run(cypher_query, query=user_query)
            return [record.data() for record in results]

    def generate_response(self, user_query: str) -> str:
        """
        Generate chatbot response using OpenAI GPT-4.
        """
        graph_results = self.query_graph(user_query)
        
        if not graph_results:
            return "I'm sorry, I couldn't find any relevant information in the knowledge graph."

        # Format results for GPT-4 prompt
        formatted_results = "\n".join(
            [f"({res['source_type']}: {res['source']}) --[{res['relationship']}]--> ({res['target_type']}: {res['target']})" for res in graph_results]
        )
        
        prompt = f"""
        You are a chatbot with access to a knowledge graph. Based on the following data:
        
        {formatted_results}
        
        Answer the user's question: "{user_query}"
        If you can't find a direct answer, provide the most relevant information from the given data.
        """
        # Use OpenAI GPT-4 to generate a response
        response = openai.ChatCompletion.create(
            model=settings.OPENAI_MODEL_ID,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
