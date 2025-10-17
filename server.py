
import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, Scalar
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    epsilons = [m.get("epsilon", 0) for _, m in metrics]
    
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "max_epsilon": max(epsilons) if epsilons else 0,
        "total_examples": sum(examples)
    }

class SecureAggregationStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_privacy_budgets = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        privacy_metrics = []
        for _, fit_res in results:
            if "epsilon" in fit_res.metrics:
                privacy_metrics.append(fit_res.metrics["epsilon"])
        
        if privacy_metrics:
            max_eps = max(privacy_metrics)
            avg_eps = np.mean(privacy_metrics)
            self.round_privacy_budgets.append(max_eps)
            
            print(f"\n{'='*60}")
            print(f"Round {server_round} Privacy Budget:")
            print(f"  Max ε across clients: {max_eps:.2f}")
            print(f"  Avg ε across clients: {avg_eps:.2f}")
            print(f"  Cumulative max ε: {sum(self.round_privacy_budgets):.2f}")
            print(f"{'='*60}\n")
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if privacy_metrics:
            aggregated_metrics["round_max_epsilon"] = max_eps
            aggregated_metrics["cumulative_epsilon"] = sum(self.round_privacy_budgets)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
        
        if accuracies:
            avg_accuracy = np.mean(accuracies)
            print(f"Round {server_round} - Average Test Accuracy: {avg_accuracy:.2f}%")
        
        return super().aggregate_evaluate(server_round, results, failures)

def start_server(num_rounds=10, min_clients=3):
    strategy = SecureAggregationStrategy(
        fraction_fit=1.0,  
        fraction_evaluate=1.0,  
        min_fit_clients=min_clients,  
        min_evaluate_clients=min_clients,  
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {
            "epochs": 1,
            "round": server_round
        },
    )
    
    print(f"""Configuration:
  Rounds: {num_rounds}
  Min Clients: {min_clients}
    """)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Total privacy budget consumed: {sum(strategy.round_privacy_budgets):.2f} ε")
    print("="*60)

if __name__ == "__main__":
    import sys
    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    min_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    start_server(num_rounds, min_clients)