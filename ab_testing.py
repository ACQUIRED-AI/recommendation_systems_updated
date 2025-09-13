# ab_testing.py
import hashlib
import json
from typing import Dict, Any
import time

class ABTestManager:
    def __init__(self):
        self.active_tests = {}
    
    def assign_user_to_variant(self, user_id: str, test_name: str) -> str:
        """Consistently assign user to A/B test variant"""
        hash_input = f"{user_id}:{test_name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        if test_name in self.active_tests:
            test_config = self.active_tests[test_name]
            variant_threshold = test_config['variant_split']
            
            if (hash_value % 100) < variant_threshold:
                return 'A'
            else:
                return 'B'
        
        return 'A'  # Default variant
    
    def log_conversion(self, user_id: str, test_name: str, conversion_type: str):
        """Log conversion event for A/B test analysis"""
        variant = self.assign_user_to_variant(user_id, test_name)
        
        # Log to analytics system
        event = {
            'user_id': user_id,
            'test_name': test_name,
            'variant': variant,
            'conversion_type': conversion_type,
            'timestamp': time.time()
        }
        
        # Send to analytics pipeline
        self._send_to_analytics(event)
    
    def _send_to_analytics(self, event: Dict):
        """Send event to analytics system"""
        # Implementation to send to your analytics system
        pass