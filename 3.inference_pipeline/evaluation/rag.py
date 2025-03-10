import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from rouge import Rouge
from db.mongo import MongoSearchConnector
from db.memgraph import MemgraphSearchConnector
from typing import List, Dict
import time

class RAGEvaluator:
    def __init__(self, ground_truth_csv: str):
        self.ground_truth = pd.read_csv(ground_truth_csv)
        self.rouge = Rouge()

    def evaluate(self, queries: List[str], mongo_connector: MongoSearchConnector, memgraph_connector: MemgraphSearchConnector) -> Dict[str, Dict[str, float]]:
        start_time = time.time()
        results = {
            "mongo": {
                "precision": [],
                "recall": [],
                "f1": [],
                "rouge_1": [],
                "rouge_2": [],
                "rouge_l": [],
                "latency": []
            },
            "memgraph": {
                "precision": [],
                "recall": [],
                "f1": [],
                "rouge_1": [],
                "rouge_2": [],
                "rouge_l": [],
                "latency": []
            }
        }

        for query in queries:
            query_start_time = time.time()
            mongo_response = mongo_connector.structured_search_index(query)
            query_end_time = time.time()
            memgraph_response = memgraph_connector.generate_response(query=query)

            ground_truth = self.ground_truth[self.ground_truth['query'] == query]['answer'].iloc[0]
            
            # Relevance metrics for MongoDB
            precision, recall, f1, _ = precision_recall_fscore_support(
                [ground_truth], [mongo_response['response']], average='weighted'
            )
            results['mongo']['precision'].append(precision)
            results['mongo']['recall'].append(recall)
            results['mongo']['f1'].append(f1)

            # ROUGE scores for MongoDB
            rouge_scores = self.rouge.get_scores(mongo_response['response'], ground_truth)
            results['mongo']['rouge_1'].append(rouge_scores[0]['rouge-1']['f'])
            results['mongo']['rouge_2'].append(rouge_scores[0]['rouge-2']['f'])
            results['mongo']['rouge_l'].append(rouge_scores[0]['rouge-l']['f'])

            # Latency for MongoDB
            results['mongo']['latency'].append(query_end_time - query_start_time)

            # Relevance metrics for Memgraph
            precision, recall, f1, _ = precision_recall_fscore_support(
                [ground_truth], [memgraph_response['response']], average='weighted'
            )
            results['memgraph']['precision'].append(precision)
            results['memgraph']['recall'].append(recall)
            results['memgraph']['f1'].append(f1)

            # ROUGE scores for Memgraph
            rouge_scores = self.rouge.get_scores(memgraph_response['response'], ground_truth)
            results['memgraph']['rouge_1'].append(rouge_scores[0]['rouge-1']['f'])
            results['memgraph']['rouge_2'].append(rouge_scores[0]['rouge-2']['f'])
            results['memgraph']['rouge_l'].append(rouge_scores[0]['rouge-l']['f'])

            # Latency for Memgraph
            results['memgraph']['latency'].append(query_end_time - query_start_time)

        end_time = time.time()
        
        # Calculate averages
        avg_results = {
            "mongo": {k: sum(v) / len(v) for k, v in results['mongo'].items()},
            "memgraph": {k: sum(v) / len(v) for k, v in results['memgraph'].items()}
        }
        avg_results['mongo']['total_evaluation_time'] = end_time - start_time
        avg_results['memgraph']['total_evaluation_time'] = end_time - start_time

        return avg_results