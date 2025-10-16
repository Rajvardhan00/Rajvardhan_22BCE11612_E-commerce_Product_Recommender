"""
E-commerce Recommender System - Streamlit Demo
File: streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommender_engine import RecommenderEngine, LLMExplainer
import sqlite3
import time

# Page configuration
st.set_page_config(
    page_title="E-commerce Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }
    .explanation-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize engine
@st.cache_resource
def load_engine():
    try:
        engine = RecommenderEngine()
        return engine
    except Exception as e:
        st.error(f"Error loading recommendation engine: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ E-commerce Recommender System</h1>', unsafe_allow_html=True)
    
    # Load engine
    engine = load_engine()
    if engine is None:
        st.error("Failed to load recommendation engine. Please check if the database exists.")
        return
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Get available users
    available_users = sorted(engine.interactions_df['user_id'].unique()[:20])
    
    # User selection
    user_id = st.sidebar.selectbox(
        "👤 Select User ID",
        options=available_users,
        index=0
    )
    
    # Recommendation type
    rec_type = st.sidebar.selectbox(
        "🎯 Recommendation Type",
        options=["Hybrid", "Collaborative Filtering", "Content-Based"],
        index=0
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "📊 Number of Recommendations",
        min_value=3,
        max_value=10,
        value=5
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎁 Personalized Recommendations")
        
        # Generate recommendations based on type
        with st.spinner("Generating recommendations..."):
            if rec_type == "Hybrid":
                recommendations = engine.hybrid_recommendations(user_id, n_recommendations)
            elif rec_type == "Collaborative Filtering":
                recommendations = engine.collaborative_filtering(user_id, n_recommendations)
            else:  # Content-Based
                recommendations = engine.content_based_filtering(user_id, n_recommendations)
            
            # Get product details
            recommended_products = engine.get_product_details(recommendations)
            
            # Get user history for explanations
            user_history = engine.get_user_history(user_id)
            explainer = LLMExplainer()
            
            # Display recommendations
            if recommended_products:
                for i, product in enumerate(recommended_products, 1):
                    with st.container():
                        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)
                        
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"### {i}. {product['name']}")
                            st.markdown(f"**Category:** {product['category']}")
                            st.markdown(f"**Price:** ${product['price']:.2f}")
                            st.markdown(f"**Rating:** {product['rating']}/5 ⭐")
                            st.markdown(f"*{product['description']}*")
                        
                        with col_b:
                            # Generate explanation
                            explanation = explainer.generate_explanation(engine, product, user_history)
                            st.markdown(f'<div class="explanation-box">💡 {explanation}</div>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this user. Try a different user or recommendation type.")
    
    with col2:
        st.subheader("📈 System Statistics")
        
        stats = engine.get_statistics()
        
        # Key metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Products", stats['total_products'])
            st.metric("Total Users", stats['total_users'])
        with col2:
            st.metric("Total Interactions", stats['total_interactions'])
            st.metric("Total Purchases", stats['total_purchases'])
        
        # User history
        st.subheader("🛒 Purchase History")
        
        purchased_ids = user_history[user_history['interaction_type'] == 'purchase']['product_id'].values
        purchased_products = engine.get_product_details(purchased_ids)
        
        if purchased_products:
            for product in purchased_products:
                with st.container():
                    st.markdown(f'<div class="product-card">', unsafe_allow_html=True)
                    st.markdown(f"**{product['name']}**")
                    st.markdown(f"${product['price']:.2f} | {product['rating']}⭐")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No purchase history for this user")
        
        # Categories distribution
        st.subheader("📊 Categories")
        for category in stats['categories']:
            st.write(f"• {category}")
    
    # Additional analytics section
    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Price distribution
        if len(engine.products_df) > 0:
            fig_price = px.histogram(engine.products_df, x='price', title='Price Distribution')
            st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Rating distribution
        if len(engine.products_df) > 0:
            fig_rating = px.histogram(engine.products_df, x='rating', title='Rating Distribution')
            st.plotly_chart(fig_rating, use_container_width=True)
    
    with col3:
        # Category distribution
        if len(engine.products_df) > 0:
            category_counts = engine.products_df['category'].value_counts()
            fig_category = px.pie(values=category_counts.values, names=category_counts.index, title='Categories')
            st.plotly_chart(fig_category, use_container_width=True)

if __name__ == "__main__":
    main()