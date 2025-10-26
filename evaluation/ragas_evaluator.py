from typing import List, Optional, Dict

from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from datasets import Dataset
import pandas as pd


class RAGASEvaluator:
    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy, context_precision]

    def evaluate_single_interaction(self,
                                    question: str,
                                    answer: str,
                                    contexts: List[str],
                                    ground_truth: Optional[str] = None) -> Dict[str, float]:
        """Evaluate a single Q&A interaction"""

        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts]
        }

        if ground_truth:
            data['ground_truth'] = [ground_truth]

        dataset = Dataset.from_dict(data)

        try:
            scores = evaluate(dataset, self.metrics)
            return {metric: scores[metric] for metric in self.metrics}
        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return {metric.__name__: 0.0 for metric in self.metrics}

    def evaluate_batch(self, interactions: List[Dict]) -> pd.DataFrame:
        """Evaluate multiple interactions"""
        results = []

        for interaction in interactions:
            scores = self.evaluate_single_interaction(
                question=interaction['question'],
                answer=interaction['answer'],
                contexts=interaction['contexts'],
                ground_truth=interaction.get('ground_truth')
            )

            results.append({
                'question': interaction['question'],
                **scores
            })

        return pd.DataFrame(results)