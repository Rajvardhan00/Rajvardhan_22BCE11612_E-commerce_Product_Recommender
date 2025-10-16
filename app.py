#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Backend API for E-commerce Recommender
Fixed version for Windows
File: app.py
"""

from flask import Flask, jsonify, request
from recommender_engine import RecommenderEngine, LLMExplainer
import json

app = Flask(__name__)

# Initialize engine (outside routes to avoid reload issues)
print("🔄 Initializing recommendation engine...")
try:
    engine = RecommenderEngine()
    explainer = LLMExplainer()
    print("✅ Engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing engine: {e}")
    engine = None
    explainer = None

@app.route('/')
def home():
    return jsonify({
        "message": "E-commerce Recommender API", 
        "status": "active",
        "endpoints": {
            "/api/recommendations/<user_id>": "Get recommendations for user",
            "/api/users": "Get available users", 
            "/api/statistics": "Get system stats",
            "/api/user/<user_id>/history": "Get user purchase history"
        }
    })

@app.route('/api/recommendations/<int:user_id>')
def get_recommendations(user_id):
    """Get recommendations for a user"""
    if engine is None:
        return jsonify({"error": "Engine not initialized", "status": "error"}), 500
        
    try:
        # Get recommendations
        recommendations = engine.hybrid_recommendations(user_id, 5)
        products = engine.get_product_details(recommendations)
        
        # Get user history for explanations
        user_history = engine.get_user_history(user_id)
        
        # Add explanations to each product
        for product in products:
            explanation = explainer.generate_explanation(engine, product, user_history)
            product['explanation'] = explanation
        
        return jsonify({
            "user_id": user_id,
            "recommendations": products,
            "count": len(products),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/users')
def get_users():
    """Get list of available users"""
    if engine is None:
        return jsonify({"error": "Engine not initialized", "status": "error"}), 500
        
    try:
        users = sorted(engine.interactions_df['user_id'].unique().tolist()[:20])
        return jsonify({
            "users": users,
            "count": len(users),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    if engine is None:
        return jsonify({"error": "Engine not initialized", "status": "error"}), 500
        
    try:
        stats = engine.get_statistics()
        stats["status"] = "success"
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/user/<int:user_id>/history')
def get_user_history(user_id):
    """Get user purchase history"""
    if engine is None:
        return jsonify({"error": "Engine not initialized", "status": "error"}), 500
        
    try:
        user_history = engine.get_user_history(user_id)
        purchases = user_history[user_history['interaction_type'] == 'purchase']
        purchased_products = engine.get_product_details(purchases['product_id'].values)
        
        return jsonify({
            "user_id": user_id,
            "purchase_history": purchased_products,
            "total_purchases": len(purchased_products),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    print("🚀 Starting E-commerce Recommender API...")
    print("📊 Available endpoints:")
    print("   http://localhost:5000/")
    print("   http://localhost:5000/api/statistics")
    print("   http://localhost:5000/api/users") 
    print("   http://localhost:5000/api/recommendations/1")
    print("   http://localhost:5000/api/user/1/history")
    print("\nPress Ctrl+C to stop the server")
    
    # Disable debug mode and reloader for Windows compatibility
    app.run(debug=False, port=5000, use_reloader=False)


# In[ ]:




