# model_serving.py
import torch
import onnx
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List

class OptimizedModelServer:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load optimized ONNX model for faster inference
        self.ort_session = ort.InferenceSession(model_path)
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up model with dummy data"""
        dummy_user = np.array([[0]], dtype=np.int64)
        dummy_item = np.array([[0]], dtype=np.int64)
        self.ort_session.run(None, {
            'user_ids': dummy_user,
            'item_ids': dummy_item
        })
    
    async def predict_batch(self, user_ids: List[int], item_ids: List[int]) -> List[float]:
        """Predict ratings for batch of user-item pairs"""
        user_array = np.array(user_ids).reshape(-1, 1).astype(np.int64)
        item_array = np.array(item_ids).reshape(-1, 1).astype(np.int64)
        
        # Run inference
        outputs = self.ort_session.run(None, {
            'user_ids': user_array,
            'item_ids': item_array
        })
        
        return outputs[0].tolist()

# Model optimization utilities
def convert_pytorch_to_onnx(pytorch_model, sample_input, output_path):
    """Convert PyTorch model to ONNX for optimized serving"""
    torch.onnx.export(
        pytorch_model,
        sample_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['user_ids', 'item_ids'],
        output_names=['predictions'],
        dynamic_axes={
            'user_ids': {0: 'batch_size'},
            'item_ids': {0: 'batch_size'},
            'predictions': {0: 'batch_size'}
        }
    )

# Load balancer for model serving
class ModelLoadBalancer:
    def __init__(self, model_servers: List[OptimizedModelServer]):
        self.servers = model_servers
        self.current_server = 0
    
    def get_next_server(self) -> OptimizedModelServer:
        """Round-robin load balancing"""
        server = self.servers[self.current_server]
        self.current_server = (self.current_server + 1) % len(self.servers)
        return server
    
    async def predict(self, user_ids: List[int], item_ids: List[int]) -> List[float]:
        server = self.get_next_server()
        return await server.predict_batch(user_ids, item_ids)