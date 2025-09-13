# streaming.py
from kafka import KafkaProducer, KafkaConsumer
import json
from typing import Dict
import asyncio
import aioredis

class RealTimeEventProcessor:
    def __init__(self, kafka_config: Dict, redis_config: Dict):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.redis_config = redis_config
    
    async def process_user_interactions(self):
        """Process real-time user interactions"""
        consumer = KafkaConsumer(
            'user_interactions',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        redis = await aioredis.create_redis_pool(
            f"redis://{self.redis_config['host']}:{self.redis_config['port']}"
        )
        
        for message in consumer:
            interaction = message.value
            
            # Update user profile in real-time
            await self.update_user_profile(redis, interaction)
            
            # Update item popularity
            await self.update_item_popularity(redis, interaction)
            
            # Trigger real-time recommendations update
            if interaction['interaction_type'] in ['purchase', 'high_rating']:
                await self.trigger_recommendation_update(interaction['user_id'])
    
    async def update_user_profile(self, redis, interaction: Dict):
        """Update user profile based on interaction"""
        user_key = f"user_profile:{interaction['user_id']}"
        
        # Update interaction counts
        await redis.hincrby(user_key, 'total_interactions', 1)
        await redis.hincrby(user_key, f"{interaction['interaction_type']}_count", 1)
        
        # Update category preferences
        category_key = f"{user_key}:categories"
        await redis.zincrby(category_key, 1, interaction['category'])
        
        # Set expiry for user data
        await redis.expire(user_key, 86400 * 30)  # 30 days
    
    async def update_item_popularity(self, redis, interaction: Dict):
        """Update item popularity scores"""
        item_key = f"item_popularity:{interaction['product_id']}"
        
        # Weight different interactions differently
        weights = {
            'view': 1,
            'click': 2,
            'cart_add': 3,
            'purchase': 5,
            'high_rating': 4
        }
        
        weight = weights.get(interaction['interaction_type'], 1)
        await redis.zincrby('popular_items', weight, interaction['product_id'])
        await redis.hincrby(item_key, 'interaction_count', 1)
    
    def send_interaction_event(self, interaction: Dict):
        """Send interaction event to Kafka"""
        self.producer.send('user_interactions', interaction)
        self.producer.flush()

# Event streaming service
class EventStreamingService:
    def __init__(self):
        self.processor = RealTimeEventProcessor(
            kafka_config={'bootstrap_servers': ['localhost:9092']},
            redis_config={'host': 'localhost', 'port': 6379}
        )
    
    async def start_processing(self):
        """Start processing real-time events"""
        await self.processor.process_user_interactions()