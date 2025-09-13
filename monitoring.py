# monitoring.py
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Metrics
RECOMMENDATION_REQUESTS = Counter('recommendation_requests_total', 'Total recommendation requests')
RECOMMENDATION_LATENCY = Histogram('recommendation_latency_seconds', 'Recommendation latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate')

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        RECOMMENDATION_REQUESTS.inc()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            RECOMMENDATION_LATENCY.observe(time.time() - start_time)
    
    return wrapper

class ModelDriftDetector:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.baseline_metrics = {}
    
    def detect_drift(self, current_metrics):
        """Detect if model performance has drifted"""
        drift_detected = False
        
        for metric, current_value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                drift_ratio = abs(current_value - baseline_value) / baseline_value
                
                if drift_ratio > self.threshold:
                    drift_detected = True
                    print(f"Drift detected in {metric}: {drift_ratio:.3f}")
        
        return drift_detected