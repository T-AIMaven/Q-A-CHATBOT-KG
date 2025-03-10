from langchain_openai import ChatOpenAI
from evaluation.rag import RAGEvaluator
from config import settings
from db.mongo import MongoSearchConnector
from db.memgraph import MemgraphSearchConnector


class LLMArangoInference:
    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.mongo_connector = MongoSearchConnector()
        self.memgraph_connector = MemgraphSearchConnector()

    def generate(
        self,
        query: str,
        enable_evaluation: bool = False,
    ) -> dict:
        mongo_response = self.mongo_connector.structured_search_index(query=query)
        memgraph_response = self.memgraph_connector.generate_response(query=query)
        response  = {
            "mongo": mongo_response,
            "memgraph": memgraph_response,
        }

        if enable_evaluation:
            evaluator = RAGEvaluator(ground_truth_csv=settings.ground_truth_csv)
            test_queries = [
                "Who is the CEO of Xngen AI?",
                "What are the main products of Xngen AI?",
                "When was Xngen AI founded?",
                # ...
            ]
            # Run the evaluation
            evaluation_results = evaluator.evaluate(test_queries, self.connector)
            response['evaluation'] = evaluation_results

        return response