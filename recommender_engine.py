#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
E-commerce Recommendation Engine
File: recommender_engine.py
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class RecommenderEngine:
    """Hybrid recommendation engine"""
    
    def __init__(self, db_name='ecommerce_recommender.db'):
        self.db_name = db_name
        self.products_df = None
        self.interactions_df = None
        self.load_data()
    
    def load_data(self):
        """Load data from database"""
        try:
            conn = sqlite3.connect(self.db_name)
            self.products_df = pd.read_sql_query("SELECT * FROM products", conn)
            self.interactions_df = pd.read_sql_query("SELECT * FROM interactions", conn)
            conn.close()
            print(f"✅ Data loaded: {len(self.products_df)} products, {len(self.interactions_df)} interactions")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Create empty DataFrames if database doesn't exist
            self.products_df = pd.DataFrame()
            self.interactions_df = pd.DataFrame()
    
    def get_user_history(self, user_id):
        """Get user interaction history"""
        if self.interactions_df is None or len(self.interactions_df) == 0:
            return pd.DataFrame()
        return self.interactions_df[self.interactions_df['user_id'] == user_id]
    
    def collaborative_filtering(self, user_id, n_recommendations=5):
        """Collaborative filtering recommendations"""
        if len(self.interactions_df) == 0:
            return []
            
        purchase_data = self.interactions_df[
            self.interactions_df['interaction_type'] == 'purchase'
        ]
        
        if len(purchase_data) == 0:
            return []
        
        user_product_matrix = purchase_data.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        
        if user_id not in user_product_matrix.index:
            return []
        
        user_similarity = cosine_similarity(user_product_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_product_matrix.index,
            columns=user_product_matrix.index
        )
        
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
        user_products = set(user_product_matrix.columns[user_product_matrix.loc[user_id] > 0])
        
        recommendations = {}
        for similar_user_id, similarity_score in similar_users.items():
            similar_user_products = set(
                user_product_matrix.columns[user_product_matrix.loc[similar_user_id] > 0]
            )
            new_products = similar_user_products - user_products
            
            for product_id in new_products:
                if product_id not in recommendations:
                    recommendations[product_id] = 0
                recommendations[product_id] += similarity_score
        
        top_products = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in top_products[:n_recommendations]]
    
    def content_based_filtering(self, user_id, n_recommendations=5):
        """Content-based filtering recommendations"""
        if len(self.products_df) == 0 or len(self.interactions_df) == 0:
            return []
            
        user_purchases = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['interaction_type'] == 'purchase')
        ]['product_id'].unique()
        
        if len(user_purchases) == 0:
            user_views = self.interactions_df[
                (self.interactions_df['user_id'] == user_id) &
                (self.interactions_df['interaction_type'] == 'view')
            ]['product_id'].unique()
            user_purchases = user_views[:3] if len(user_views) > 0 else [1, 2, 3]
        
        self.products_df['content'] = (
            self.products_df['category'] + ' ' + 
            self.products_df['description']
        )
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.products_df['content'])
        content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        recommendations = set()
        for product_id in user_purchases:
            product_idx = product_id - 1
            similar_indices = content_similarity[product_idx].argsort()[-6:-1][::-1]
            
            for idx in similar_indices:
                similar_product_id = self.products_df.iloc[idx]['id']
                if similar_product_id not in user_purchases:
                    recommendations.add(similar_product_id)
        
        return list(recommendations)[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=5):
        """Hybrid recommendations (60% collab, 40% content)"""
        collab_recs = self.collaborative_filtering(user_id, n_recommendations)
        content_recs = self.content_based_filtering(user_id, n_recommendations)
        
        all_recs = {}
        for i, product_id in enumerate(collab_recs):
            all_recs[product_id] = (len(collab_recs) - i) * 0.6
        
        for i, product_id in enumerate(content_recs):
            if product_id in all_recs:
                all_recs[product_id] += (len(content_recs) - i) * 0.4
            else:
                all_recs[product_id] = (len(content_recs) - i) * 0.4
        
        sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_recs[:n_recommendations]]
    
    def get_product_details(self, product_ids):
        """Get detailed information for given product IDs"""
        # FIXED: Proper check for empty array
        if product_ids is None or len(product_ids) == 0:
            return []
        
        products = self.products_df[self.products_df['id'].isin(product_ids)]
        return products.to_dict('records')
    
    def get_statistics(self):
        """Get system statistics"""
        if len(self.products_df) == 0 or len(self.interactions_df) == 0:
            return {
                'total_products': 0,
                'total_users': 0,
                'total_interactions': 0,
                'total_purchases': 0,
                'categories': [],
                'avg_rating': 0,
                'price_range': {'min': 0, 'max': 0, 'avg': 0}
            }
            
        return {
            'total_products': len(self.products_df),
            'total_users': self.interactions_df['user_id'].nunique(),
            'total_interactions': len(self.interactions_df),
            'total_purchases': len(self.interactions_df[
                self.interactions_df['interaction_type'] == 'purchase'
            ]),
            'categories': self.products_df['category'].unique().tolist(),
            'avg_rating': float(self.products_df['rating'].mean()),
            'price_range': {
                'min': float(self.products_df['price'].min()),
                'max': float(self.products_df['price'].max()),
                'avg': float(self.products_df['price'].mean())
            }
        }


class LLMExplainer:
    """Generate explanations for recommendations"""
    
    def __init__(self):
        self.fallback_mode = True
    
    def generate_explanation(self, recommender_engine, product, user_history):
        """Generate explanation"""
        purchased_products = user_history[
            user_history['interaction_type'] == 'purchase'
        ]['product_id'].values
        
        purchased_names = []
        if len(purchased_products) > 0:
            purchased_df = recommender_engine.products_df[
                recommender_engine.products_df['id'].isin(purchased_products)
            ]
            purchased_names = purchased_df['name'].tolist()
        
        return self._generate_smart_explanation(product, purchased_names)
    
    def _generate_smart_explanation(self, product, purchased_names):
        """Generate smart explanation"""
        if purchased_names:
            categories = set()
            category_keywords = {
                'Electronics': ['headphone', 'mouse', 'charger', 'laptop', 'phone'],
                'Sports & Fitness': ['yoga', 'fitness', 'sport', 'gym', 'exercise'],
                'Books': ['book', 'novel', 'planner'],
                'Home & Kitchen': ['kitchen', 'pan', 'coffee', 'bottle', 'knife'],
                'Fashion': ['shoes', 'jacket', 'backpack', 'glasses']
            }
            
            for name in purchased_names:
                name_lower = name.lower()
                for category, keywords in category_keywords.items():
                    if any(keyword in name_lower for keyword in keywords):
                        categories.add(category)
            
            if categories:
                category_text = " and ".join(categories)
                return (f"Based on your interest in {category_text}, we recommend this "
                       f"{product['name']}. It has a {product['rating']}/5 rating and offers "
                       f"excellent value at ${product['price']:.2f}.")
        
        return (f"We think you'll love this {product['name']}! "
               f"It's highly rated ({product['rating']}/5) in the {product['category']} category.")


if __name__ == "__main__":
    print("Testing recommender engine...")
    try:
        engine = RecommenderEngine()
        print("✅ Engine loaded successfully!")
        
        # Test the problematic method
        test_ids = np.array([1, 2, 3])
        result = engine.get_product_details(test_ids)
        print(f"✅ get_product_details test passed: {len(result)} products")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# In[ ]:




