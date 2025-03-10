import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from rouge import Rouge
from db.ddb import StructuredSearchConnector
from typing import List, Dict
import time

class RAGEvaluator:
    def __init__(self, ground_truth_csv: str):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.rouge = Rouge()

    def evaluate(self, queries: List[str], rag_system: StructuredSearchConnector) -> Dict[str, float]:
        start_time = time.time()
        results = {
            "precision": [],
            "recall": [],
            "f1": [],
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
            "latency": []
        }

        for query in queries:
            query_start_time = time.time()
            rag_response = rag_system.structured_search_index(query)
            query_end_time = time.time()

            ground_truth = self.ground_truth[self.ground_truth['query'] == query]['answer'].iloc[0]
            
            # Relevance metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                [ground_truth], [rag_response['response']], average='weighted'
            )
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)

            # ROUGE scores for generated text quality
            rouge_scores = self.rouge.get_scores(rag_response['response'], ground_truth)
            results['rouge_1'].append(rouge_scores[0]['rouge-1']['f'])
            results['rouge_2'].append(rouge_scores[0]['rouge-2']['f'])
            results['rouge_l'].append(rouge_scores[0]['rouge-l']['f'])

            # Latency
            results['latency'].append(query_end_time - query_start_time)

        end_time = time.time()
        
        # Calculate averages
        avg_results = {k: sum(v) / len(v) for k, v in results.items()}
        avg_results['total_evaluation_time'] = end_time - start_time

        return avg_results

# def main():
#     # Initialize the RAG system
#     rag_system = StructuredSearchConnector()

#     # Initialize the evaluator with ground truth data
#     evaluator = RAGEvaluator("ground_truth.csv")

#     # Define a list of test queries
#     test_queries = [
#         "Who is the CEO of Xngen AI?",
#         "What are the main products of Xngen AI?",
#         "When was Xngen AI founded?",
#         # Add more test queries here
#     ]

#     # Run the evaluation
#     evaluation_results = evaluator.evaluate(test_queries, rag_system)

#     # Print the results
#     print("Evaluation Results:")
#     for metric, value in evaluation_results.items():
#         print(f"{metric}: {value}")

# if __name__ == "__main__":
#     main()