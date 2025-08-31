import os
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from django.shortcuts import render
from django.http import HttpResponse
from typing import List

# LangChain imports
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

def generate_sample_ecommerce_data():
    """Generate 1000 sample ecommerce products and save to JSON"""
    
    categories = [
        "Electronics", "Clothing", "Home & Garden", "Sports & Outdoors", 
        "Books", "Health & Beauty", "Toys & Games", "Automotive", 
        "Jewelry", "Food & Beverages", "Pet Supplies", "Office Supplies"
    ]
    
    brands = [
        "TechPro", "StyleMax", "ComfortHome", "ActiveLife", "SmartChoice", 
        "PremiumBrand", "EcoFriendly", "BudgetBest", "LuxuryLine", "QuickFix",
        "MegaBrand", "TopTier", "ValuePlus", "EliteChoice", "ProGear"
    ]
    
    # Product templates by category
    product_templates = {
        "Electronics": [
            "Wireless Bluetooth Headphones", "Smart TV", "Laptop Computer", 
            "Smartphone", "Tablet", "Gaming Console", "Smart Watch", 
            "Wireless Speaker", "Camera", "Power Bank", "Router", "Monitor"
        ],
        "Clothing": [
            "T-Shirt", "Jeans", "Dress", "Jacket", "Sneakers", "Hoodie", 
            "Shorts", "Sweater", "Boots", "Skirt", "Blouse", "Pants"
        ],
        "Home & Garden": [
            "Coffee Maker", "Vacuum Cleaner", "Air Purifier", "Desk Lamp", 
            "Garden Hose", "Tool Set", "Bedding Set", "Dining Table", 
            "Plant Pot", "Wall Clock", "Storage Box", "Kitchen Scale"
        ],
        "Sports & Outdoors": [
            "Yoga Mat", "Running Shoes", "Bicycle", "Camping Tent", 
            "Fitness Tracker", "Water Bottle", "Dumbbells", "Backpack", 
            "Soccer Ball", "Tennis Racket", "Hiking Boots", "Swim Goggles"
        ],
        "Books": [
            "Fiction Novel", "Self-Help Book", "Cookbook", "Biography", 
            "Science Book", "History Book", "Art Book", "Travel Guide", 
            "Children's Book", "Poetry Collection", "Technical Manual", "Dictionary"
        ],
        "Health & Beauty": [
            "Skincare Set", "Hair Dryer", "Electric Toothbrush", "Perfume", 
            "Makeup Kit", "Vitamins", "Face Mask", "Shampoo", "Moisturizer", 
            "Nail Polish", "Sunscreen", "Body Lotion"
        ]
    }
    
    products = []
    reviews_data = []
    
    for i in range(1, 1001):
        category = random.choice(categories)
        brand = random.choice(brands)
        product_name = random.choice(product_templates.get(category, ["Generic Product"]))
        
        # Generate product
        product = {
            "id": i,
            "name": f"{brand} {product_name}",
            "description": f"High-quality {product_name.lower()} from {brand}. Perfect for everyday use with excellent performance and durability.",
            "category": category,
            "brand": brand,
            "price": round(random.uniform(9.99, 999.99), 2),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "stock_quantity": random.randint(0, 500),
            "is_active": random.choice([True, True, True, False]),  # 75% active
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
            "tags": random.sample(["bestseller", "eco-friendly", "premium", "budget", "new-arrival", "sale"], k=random.randint(1, 3))
        }
        products.append(product)
        
        # Generate 1-5 reviews per product
        for review_id in range(1, random.randint(2, 6)):
            review = {
                "id": len(reviews_data) + 1,
                "product_id": i,
                "product_name": product["name"],
                "rating": random.randint(1, 5),
                "review_text": random.choice([
                    "Great product! Highly recommended.",
                    "Good value for money. Works as expected.",
                    "Excellent quality and fast shipping.",
                    "Not bad, but could be better.",
                    "Amazing product! Exceeded my expectations.",
                    "Decent product for the price.",
                    "Love it! Will buy again.",
                    "Good build quality and design.",
                    "Fair product, nothing special.",
                    "Outstanding! Worth every penny."
                ]),
                "reviewer_name": f"Customer{random.randint(1000, 9999)}",
                "created_at": (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat(),
                "verified_purchase": random.choice([True, False])
            }
            reviews_data.append(review)
    
    # Generate sample orders
    orders_data = []
    for order_id in range(1, 201):  # 200 orders
        num_items = random.randint(1, 5)
        order_items = []
        total_amount = 0
        
        for item_num in range(num_items):
            product = random.choice(products)
            quantity = random.randint(1, 3)
            item_price = product["price"]
            
            order_item = {
                "product_id": product["id"],
                "product_name": product["name"],
                "quantity": quantity,
                "price": item_price,
                "total": round(quantity * item_price, 2)
            }
            order_items.append(order_item)
            total_amount += order_item["total"]
        
        order = {
            "id": order_id,
            "user_id": random.randint(1, 100),
            "items": order_items,
            "total_amount": round(total_amount, 2),
            "status": random.choice(["pending", "processing", "shipped", "delivered", "cancelled"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        }
        orders_data.append(order)
    
    # Create the complete dataset
    ecommerce_data = {
        "products": products,
        "reviews": reviews_data,
        "orders": orders_data,
        "generated_at": datetime.now().isoformat(),
        "total_products": len(products),
        "total_reviews": len(reviews_data),
        "total_orders": len(orders_data)
    }
    
    # Save to JSON file
    with open('rag/ecommerce_data.json', 'w') as f:
        json.dump(ecommerce_data, f, indent=2)
    
    print(f"Generated {len(products)} products, {len(reviews_data)} reviews, and {len(orders_data)} orders")
    return ecommerce_data

def load_ecommerce_data_from_json():
    """Load ecommerce data from JSON file and convert to documents"""
    try:
        # Check if file exists, if not generate it
        if not os.path.exists('rag/ecommerce_data.json'):
            print("Generating sample ecommerce data...")
            generate_sample_ecommerce_data()
        
        # Load the JSON data
        with open('rag/ecommerce_data.json', 'r') as f:
            data = json.load(f)
        
        all_documents = []
        
        # Process products
        for product in data['products']:
            content = f"""
            Product: {product['name']}
            Category: {product['category']}
            Brand: {product['brand']}
            Price: ${product['price']}
            Rating: {product['rating']}/5.0
            Description: {product['description']}
            Stock: {product['stock_quantity']} units available
            Status: {'Active' if product['is_active'] else 'Inactive'}
            Tags: {', '.join(product['tags'])}
            Added: {product['created_at'][:10]}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    'source': 'products',
                    'id': product['id'],
                    'category': product['category'],
                    'brand': product['brand'],
                    'price': product['price'],
                    'rating': product['rating']
                }
            )
            all_documents.append(doc)
        
        # Process reviews
        for review in data['reviews']:
            content = f"""
            Product Review: {review['product_name']}
            Rating: {review['rating']}/5 stars
            Review: {review['review_text']}
            Reviewer: {review['reviewer_name']}
            Verified Purchase: {'Yes' if review['verified_purchase'] else 'No'}
            Date: {review['created_at'][:10]}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    'source': 'reviews',
                    'product_id': review['product_id'],
                    'rating': review['rating'],
                    'verified': review['verified_purchase']
                }
            )
            all_documents.append(doc)
        
        # Process recent orders (last 30 days)
        for order in data['orders']:
            for item in order['items']:
                content = f"""
                Recent Order Information:
                Product: {item['product_name']}
                Quantity Ordered: {item['quantity']}
                Price: ${item['price']} each
                Order Total: ${item['total']}
                Order Status: {order['status']}
                Order Date: {order['created_at'][:10]}
                Customer ID: {order['user_id']}
                """
                
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        'source': 'orders',
                        'order_id': order['id'],
                        'product_id': item['product_id'],
                        'status': order['status']
                    }
                )
                all_documents.append(doc)
        
        print(f"Loaded {len(all_documents)} documents from JSON data")
        return all_documents
        
    except Exception as e:
        print(f"Error loading data from JSON: {e}")
        return []

def setup_rag_chain():
    """Set up RAG system with JSON data"""
    try:
        # Load documents from JSON
        docs = load_ecommerce_data_from_json()
        
        if not docs:
            print("No documents loaded from JSON file")
            return None
        
        print(f"Setting up RAG with {len(docs)} documents")
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Set up the DeepSeek model
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
        
        # Enhanced prompt template for ecommerce
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI shopping assistant for an ecommerce platform. Answer questions about products, reviews, and orders based on the provided context.
            
            You can help with:
            - Product recommendations and comparisons
            - Pricing and availability information
            - Customer reviews and ratings
            - Product categories and brands
            - Recent order trends
            
            Always be helpful, accurate, and specific. If you don't have exact information, say so clearly.
            When recommending products, consider price, rating, and availability.
            
            Context: {context}
            """),
            ("user", "{input}"),
        ])
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 8}  # Retrieve more documents for better context
        )
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        print(f"Error setting up RAG chain: {e}")
        return None

# Global variable to store the RAG chain
rag_chain = None

def initialize_rag():
    """Initialize RAG chain"""
    global rag_chain
    rag_chain = setup_rag_chain()
    return rag_chain is not None

def refresh_rag_data():
    """Regenerate data and refresh RAG system"""
    global rag_chain
    try:
        # Generate new data
        generate_sample_ecommerce_data()
        # Reinitialize RAG chain
        rag_chain = setup_rag_chain()
        return rag_chain is not None
    except Exception as e:
        print(f"Error refreshing data: {e}")
        return False

def rag_view(request):
    """Main view for RAG application"""
    global rag_chain
    
    response = None
    error = None
    
    # Initialize RAG chain if not already done
    if rag_chain is None:
        if not initialize_rag():
            error = "Failed to initialize the RAG system. Please check the logs."
    
    if request.method == 'POST':
        action = request.POST.get('action', 'query')
        
        if action == 'refresh':
            if refresh_rag_data():
                response = "✅ Successfully generated new sample data and refreshed the system!"
            else:
                error = "❌ Failed to refresh data. Please check the logs."
        
        elif action == 'query':
            query = request.POST.get('query', '').strip()
            if query and rag_chain:
                try:
                    result = rag_chain.invoke({"input": query})
                    response = result['answer']
                except Exception as e:
                    error = f"An error occurred while processing your query: {e}"
            else:
                error = "Please enter a valid query."
    
    # Get some stats to display
    stats = {}
    try:
        if os.path.exists('rag/ecommerce_data.json'):
            with open('rag/ecommerce_data.json', 'r') as f:
                data = json.load(f)
                stats = {
                    'total_products': data.get('total_products', 0),
                    'total_reviews': data.get('total_reviews', 0),
                    'total_orders': data.get('total_orders', 0),
                    'generated_at': data.get('generated_at', 'Unknown')[:19].replace('T', ' ')
                }
    except:
        pass

    return render(request, 'rag/rag.html', {
        'response': response,
        'error': error,
        'stats': stats
    })

