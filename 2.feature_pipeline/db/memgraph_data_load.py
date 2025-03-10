from neo4j import GraphDatabase
import pandas as pd
import os
import openai
from typing import List, Dict

# Set up OpenAI API Key (Replace with your actual key)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Memgraph connection details
MEMGRAPH_URI = "bolt://localhost:7687"
AUTH = ("", "")  # No authentication by default for Memgraph

class MemgraphDataLoadConnector:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def import_csv_to_graph(self, items: list[dict]):
        """
        Import CSV data into Memgraph as nodes and relationships.
        """
        df = pd.DataFrame(items)
        columns = df.columns.tolist()

        with self.driver.session() as session:
            # Create a node for each unique value in each column
            for column in columns:
                query = f"""
                UNWIND $values AS value
                MERGE (n:{column} {{value: value}})
                """
                session.run(query, values=df[column].unique().tolist())

            # Create relationships between nodes in adjacent columns
            for i in range(len(columns) - 1):
                col1, col2 = columns[i], columns[i+1]
                query = f"""
                MATCH (n:{col1}), (m:{col2})
                WHERE n.value = $val1 AND m.value = $val2
                MERGE (n)-[r:RELATED_TO]->(m)
                """
                for _, row in df.iterrows():
                    session.run(query, val1=row[col1], val2=row[col2])

        print("CSV data imported into Memgraph.")

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
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        
        return response['choices'][0]['message']['content']

def main():
    # Initialize chatbot
    chatbot = MemgraphChatbot(MEMGRAPH_URI, AUTH)

    try:
        # Import CSV data into Memgraph (replace 'data.csv' with your file path)
        chatbot.import_csv_to_graph("data.csv")

        print("Chatbot is ready! Type your questions below (type 'exit' to quit).")

        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break

            # Generate response from the chatbot
            response = chatbot.generate_response(user_input)
            print(f"Chatbot: {response}")

    finally:
        chatbot.close()
