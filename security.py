# security.py
import hashlib
import hmac
from cryptography.fernet import Fernet
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any, Mapping
import numpy as np
from fastapi import HTTPException

class SecurityManager:
    def __init__(self, secret_key: str, encryption_key: bytes):
        self.secret_key = secret_key
        self.fernet = Fernet(encryption_key)
        self.token_expiry_hours = 24
    
    def hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy protection"""
        return hashlib.sha256(f"{user_id}{self.secret_key}".encode()).hexdigest()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive user data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive user data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_api_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for API access"""
        payload = {
            'user_id': self.hash_user_id(user_id),
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_api_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Privacy-preserving recommendation techniques
class PrivacyPreservingRecommender:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Differential privacy parameter
    
    def add_differential_privacy_noise(self, recommendations: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Add differential privacy noise to recommendations"""
        noisy_recommendations = []
        
        for product_id, score in recommendations:
            # Add Laplace noise for differential privacy
            noise = np.random.laplace(0, 1.0 / self.epsilon)
            noisy_score = max(0, min(5, score + noise))  # Clip to valid range
            noisy_recommendations.append((product_id, noisy_score))
        
        # Re-sort by noisy scores
        noisy_recommendations.sort(key=lambda x: x[1], reverse=True)
        return noisy_recommendations
    
    def federated_learning_update(self, local_gradients: List[np.ndarray], 
                                client_weights: List[float]) -> np.ndarray:
        """Aggregate model updates using federated learning"""
        # Weighted average of gradients
        aggregated_gradient = np.zeros_like(local_gradients[0])
        
        total_weight = sum(client_weights)
        for gradient, weight in zip(local_gradients, client_weights):
            aggregated_gradient += (weight / total_weight) * gradient
        
        # Add differential privacy noise
        noise_scale = 1.0 / (self.epsilon * total_weight)
        noise = np.random.laplace(0, noise_scale, aggregated_gradient.shape)
        
        return aggregated_gradient + noise

# Rate limiting and API security
from functools import wraps
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed based on rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[identifier] = [req_time for req_time in self.requests[identifier] 
                                   if req_time > minute_ago]
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True

def rate_limit(requests_per_minute: int = 60):
    limiter = RateLimiter(requests_per_minute)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user identifier from request
            request = kwargs.get('request') or args[0]
            user_id = getattr(request, 'user_id', request.client.host)
            
            if not limiter.is_allowed(user_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator