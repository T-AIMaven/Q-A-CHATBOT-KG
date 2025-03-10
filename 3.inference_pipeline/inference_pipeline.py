import pandas as pd
from langchain_openai import ChatOpenAI
from llm.chain import GeneralChain
from evaluation.rag import RAGEvaluator
from config import settings
from db.ddb import StructuredSearchConnector

class LLMArangoML:
    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self.chain = GeneralChain().get_chain(
            llm=self.model, output_key="answer"
        )
        self.connector = StructuredSearchConnector()

    def generate(
        self,
        query: str,
        mechanism: str,
        enable_evaluation: bool = False,
        enable_monitoring: bool = True,
    ) -> dict:
        response = self.connector.structured_search_index(query=query)

        if enable_evaluation:
            evaluator = RAGEvaluator(ground_truth_csv="1.datacollection_pipeline/datasets/ground_truth.csv")
            
            # Define a list of test queries
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