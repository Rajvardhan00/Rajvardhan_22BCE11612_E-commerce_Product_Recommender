E-commerce Product Recommender System
Project Overview
A hybrid recommendation system that combines collaborative filtering and content-based filtering with AI-powered explanations. The system provides personalized product recommendations with natural language explanations for why each product is suggested.

Features
Hybrid recommendation algorithm combining collaborative and content-based filtering

AI-generated explanations for each recommendation

RESTful API backend built with Flask

Interactive web dashboard built with Streamlit

SQLite database for product catalog and user interactions

Project Structure
app.py - Flask backend API

streamlit_app.py - Streamlit frontend dashboard

recommender_engine.py - Core recommendation engine

ecommerce_recommender.db - SQLite database

requirements.txt - Python dependencies

LLM_Assignment.ipynb - Development notebook

Installation
Install required packages:

text
pip install -r requirements.txt
The system requires:

Python 3.8+

Flask

Streamlit

Pandas

Scikit-learn

Plotly

Usage
Starting the Backend API
Run the Flask API server:

text
python app.py
The API will be available at: http://localhost:5000

Starting the Frontend Dashboard
Run the Streamlit dashboard:

text
streamlit run streamlit_app.py
The dashboard will be available at: http://localhost:8501

API Endpoints
GET / - API information

GET /api/statistics - System statistics and metrics

GET /api/users - List of available users

GET /api/recommendations/<user_id> - Personalized recommendations for user

GET /api/user/<user_id>/history - User purchase history

Technical Implementation
The recommendation system uses a hybrid approach:

Collaborative Filtering: Finds users with similar purchase patterns

Content-Based Filtering: Recommends products similar to user's previous interests

Hybrid Combination: Weighted combination of both methods

AI Explanations: Generates natural language explanations based on user behavior and product features

The system includes a sample database with 20 products and 629 user interactions across multiple categories including Electronics, Sports, Books, and Home products.

Evaluation Features
Recommendation accuracy through hybrid algorithm

Quality of AI-generated explanations

Clean code design and architecture

Comprehensive documentation

Working API and interactive dashboard

