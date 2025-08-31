import os
from dotenv import load_dotenv
from django.shortcuts import render
from django.http import HttpResponse
import psycopg2
import pandas as pd
from typing import List

# LangChain imports
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

# Load environment variables
load_dotenv()
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'ecommerce_db'),
    'user': os.getenv('DB_USER', 'your_username'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'port': os.getenv('DB_PORT', '5432')
}

def load_ecommerce_data():
    """Load data from PostgreSQL database and convert to documents"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DATABASE_CONFIG)
        
        # Define your queries for different ecommerce tables
        queries = {
            'products': """
                SELECT 
                    p.id,
                    p.name,
                    p.description,
                    p.price,
                    c.name as category,
                    p.stock_quantity,
                    p.brand,
                    p.rating
                FROM products p
                LEFT JOIN categories c ON p.category_id = c.id
                WHERE p.is_active = true
            """,
            
            'categories': """
                SELECT 
                    id,
                    name,
                    description,
                    parent_category
                FROM categories
                WHERE is_active = true
            """,
            
            'reviews': """
                SELECT 
                    r.id,
                    r.product_id,
                    p.name as product_name,
                    r.rating,
                    r.review_text,
                    r.created_at
                FROM reviews r
                JOIN products p ON r.product_id = p.id
                WHERE r.is_approved = true
                ORDER BY r.created_at DESC
                LIMIT 1000
            """,
            
            'orders': """
                SELECT 
                    o.id,
                    o.user_id,
                    o.total_amount,
                    o.status,
                    o.created_at,
                    oi.product_id,
                    p.name as product_name,
                    oi.quantity,
                    oi.price
                FROM orders o
                JOIN order_items oi ON o.id = oi.order_id
                JOIN products p ON oi.product_id = p.id
                WHERE o.created_at >= CURRENT_DATE - INTERVAL '30 days'
            """
        }
        
        all_documents = []
        
        for table_name, query in queries.items():
            df = pd.read_sql_query(query, conn)
            
            # Convert each row to a document
            for _, row in df.iterrows():
                if table_name == 'products':
                    content = f"""
                    Product: {row['name']}
                    Description: {row['description']}
                    Price: ${row['price']}
                    Category: {row['category']}
                    Brand: {row['brand']}
                    Rating: {row['rating']}/5
                    Stock: {row['stock_quantity']} units
                    """
                    
                elif table_name == 'categories':
                    content = f"""
                    Category: {row['name']}
                    Description: {row['description']}
                    Parent Category: {row['parent_category']}
                    """
                    
                elif table_name == 'reviews':
                    content = f"""
                    Product Review for: {row['product_name']}
                    Rating: {row['rating']}/5
                    Review: {row['review_text']}
                    Date: {row['created_at']}
                    """
                    
                elif table_name == 'orders':
                    content = f"""
                    Order Information:
                    Product: {row['product_name']}
                    Quantity: {row['quantity']}
                    Price: ${row['price']}
                    Order Status: {row['status']}
                    Order Date: {row['created_at']}
                    """
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': table_name,
                        'id': row['id'],
                        **{k: v for k, v in row.items() if k != 'id'}
                    }
                )
                all_documents.append(doc)
        
        conn.close()
        return all_documents
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return []

def setup_rag_chain():
    """Set up RAG system with PostgreSQL data"""
    try:
        # Load documents from database
        docs = load_ecommerce_data()
        
        if not docs:
            print("No documents loaded from database")
            return None
        
        print(f"Loaded {len(docs)} documents from database")
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Set up the DeepSeek model
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
        
        # Enhanced prompt template for ecommerce
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant for an ecommerce platform. Answer the user's question based on the following context about products, categories, reviews, and orders.
            
            Provide helpful, accurate information about:
            - Product details, prices, and availability
            - Product categories and recommendations
            - Customer reviews and ratings
            - Order information and trends
            
            If you don't have specific information, say so clearly.
            
            Context: {context}
            """),
            ("user", "{input}"),
        ])
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return None

# Global variable to store the RAG chain
rag_chain = None

def initialize_rag():
    """Initialize RAG chain (call this when Django starts)"""
    global rag_chain
    rag_chain = setup_rag_chain()

def refresh_rag_data():
    """Refresh RAG system with latest database data"""
    global rag_chain
    rag_chain = setup_rag_chain()
    return rag_chain is not None

def rag_view(request):
    """Main view for RAG application"""
    global rag_chain
    
    response = None
    error = None
    
    # Initialize RAG chain if not already done
    if rag_chain is None:
        initialize_rag()
    
    if request.method == 'POST':
        action = request.POST.get('action', 'query')
        
        if action == 'refresh':
            # Refresh data from database
            if refresh_rag_data():
                response = "Data refreshed successfully from database!"
            else:
                error = "Failed to refresh data from database."
        
        elif action == 'query':
            query = request.POST.get('query', '')
            if query and rag_chain:
                try:
                    result = rag_chain.invoke({"input": query})
                    response = result['answer']
                    
                    # Optionally include source information
                    sources = []
                    if 'context' in result:
                        for doc in result['context']:
                            source_info = {
                                'source': doc.metadata.get('source', 'Unknown'),
                                'id': doc.metadata.get('id', 'N/A')
                            }
                            sources.append(source_info)
                    
                except Exception as e:
                    error = f"An error occurred: {e}"
            else:
                error = "Please enter a query and ensure the system is properly initialized."

    return render(request, 'rag/rag.html', {
        'response': response,
        'error': error
    })

# Additional utility functions

def get_product_recommendations(user_query: str, limit: int = 5):
    """Get product recommendations based on user query"""
    if not rag_chain:
        return []
    
    try:
        result = rag_chain.invoke({
            "input": f"Recommend {limit} products based on: {user_query}"
        })
        return result['answer']
    except Exception as e:
        return f"Error getting recommendations: {e}"

def search_products_by_category(category: str):
    """Search for products in a specific category"""
    if not rag_chain:
        return "RAG system not initialized"
    
    try:
        result = rag_chain.invoke({
            "input": f"Show me all products in the {category} category"
        })
        return result['answer']
    except Exception as e:
        return f"Error searching category: {e}"