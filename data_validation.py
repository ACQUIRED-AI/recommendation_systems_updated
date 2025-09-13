# data_validation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

class DataQualityChecker:
    def __init__(self):
        self.quality_rules = {
            'missing_values_threshold': 0.1,
            'duplicate_threshold': 0.05,
            'rating_range': (1.0, 5.0),
            'price_range': (0.1, 10000.0)
        }
    
    def validate_interactions_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate user interactions data quality"""
        issues = []
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > self.quality_rules['missing_values_threshold']:
            issues.append(f"High missing values ratio: {missing_ratio:.3f}")
        
        # Check for duplicates
        duplicate_ratio = df.duplicated().sum() / len(df)
        if duplicate_ratio > self.quality_rules['duplicate_threshold']:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.3f}")
        
        # Validate rating range
        if 'rating' in df.columns:
            invalid_ratings = df[
                (df['rating'] < self.quality_rules['rating_range'][0]) |
                (df['rating'] > self.quality_rules['rating_range'][1])
            ]
            if len(invalid_ratings) > 0:
                issues.append(f"Invalid ratings found: {len(invalid_ratings)} records")
        
        # Check for temporal anomalies
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            future_dates = df[df['timestamp'] > datetime.now()]
            if len(future_dates) > 0:
                issues.append(f"Future dates found: {len(future_dates)} records")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_records': len(df),
            'validation_timestamp': datetime.now()
        }
    
    def validate_products_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate products data quality"""
        issues = []
        
        # Check required fields
        required_fields = ['product_id', 'name', 'price_current', 'category']
        for field in required_fields:
            if field not in df.columns:
                issues.append(f"Missing required field: {field}")
            elif df[field].isnull().any():
                issues.append(f"Null values in required field: {field}")
        
        # Validate price range
        if 'price_current' in df.columns:
            invalid_prices = df[
                (df['price_current'] < self.quality_rules['price_range'][0]) |
                (df['price_current'] > self.quality_rules['price_range'][1])
            ]
            if len(invalid_prices) > 0:
                issues.append(f"Invalid prices found: {len(invalid_prices)} records")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_records': len(df)
        }

# Data drift detection
class DataDriftDetector:
    def __init__(self):
        self.reference_stats = {}
    
    def set_reference_data(self, df: pd.DataFrame):
        """Set reference statistics for drift detection"""
        self.reference_stats = {
            'mean_rating': df['rating'].mean() if 'rating' in df.columns else None,
            'category_distribution': df['category'].value_counts(normalize=True).to_dict(),
            'price_mean': df['price_current'].mean() if 'price_current' in df.columns else None,
            'interactions_per_user': df.groupby('user_id').size().mean() if 'user_id' in df.columns else None
        }
    
    def detect_drift(self, new_df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, any]:
        """Detect if new data has drifted from reference"""
        drift_detected = False
        drift_details = {}
        
        # Check rating drift
        if 'rating' in new_df.columns and self.reference_stats['mean_rating']:
            new_mean_rating = new_df['rating'].mean()
            rating_drift = abs(new_mean_rating - self.reference_stats['mean_rating']) / self.reference_stats['mean_rating']
            
            if rating_drift > threshold:
                drift_detected = True
                drift_details['rating_drift'] = {
                    'reference': self.reference_stats['mean_rating'],
                    'current': new_mean_rating,
                    'drift_ratio': rating_drift
                }
        
        # Check category distribution drift
        if 'category' in new_df.columns:
            new_category_dist = new_df['category'].value_counts(normalize=True).to_dict()
            
            # Calculate JS divergence for distribution comparison
            categories = set(list(self.reference_stats['category_distribution'].keys()) + 
                           list(new_category_dist.keys()))
            
            ref_probs = [self.reference_stats['category_distribution'].get(cat, 0) for cat in categories]
            new_probs = [new_category_dist.get(cat, 0) for cat in categories]
            
            js_divergence = self._js_divergence(ref_probs, new_probs)
            
            if js_divergence > threshold:
                drift_detected = True
                drift_details['category_drift'] = {
                    'js_divergence': js_divergence,
                    'reference_dist': self.reference_stats['category_distribution'],
                    'current_dist': new_category_dist
                }
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'check_timestamp': datetime.now()
        }
    
    def _js_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate Jensen-Shannon divergence"""
        p = np.array(p)
        q = np.array(q)
        m = (p + q) / 2
        
        return (self._kl_divergence(p, m) + self._kl_divergence(q, m)) / 2
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate KL divergence"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        return np.sum(p * np.log(p / q))