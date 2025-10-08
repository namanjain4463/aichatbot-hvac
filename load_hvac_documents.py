#!/usr/bin/env python3
"""
Script to load HVAC documents into the chatbot
This script demonstrates how to programmatically load HVAC documents
"""

import os
import glob
from hvac_code_chatbot import HVACCodeProcessor, HVACGraphRAG
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_hvac_documents_from_folder(folder_path: str):
    """Load all HVAC documents from a folder and process them"""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} does not exist!")
        return
    
    print(f"🏗️ Loading HVAC documents from {folder_path}...")
    
    # Initialize processor
    processor = HVACCodeProcessor()
    
    # Load documents
    documents = processor.load_hvac_documents(folder_path)
    
    if not documents:
        print("❌ No documents found!")
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create chunks
    print("📝 Creating document chunks...")
    chunks = processor.create_hvac_chunks(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Create vector store
    print("🔍 Creating vector embeddings...")
    vector_store = FAISS.from_documents(chunks, processor.embeddings)
    print("✅ Vector store created")
    
    # Create knowledge graph with credentials from environment variables
    print("🕸️ Creating knowledge graph...")
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASS")
    
    graphrag = HVACGraphRAG(neo4j_uri, neo4j_user, neo4j_pass)
    entities = graphrag.create_hvac_entities(documents)
    graphrag.create_knowledge_graph(entities, documents)
    print(f"✅ Created knowledge graph with {len(entities)} entities")
    
    # Save vector store
    vector_store.save_local("hvac_vector_store")
    print("✅ Vector store saved locally")
    
    print("\n🎉 HVAC documents processed successfully!")
    print("You can now run the chatbot with: streamlit run hvac_code_chatbot.py")

if __name__ == "__main__":
    # Load documents from hvac_documents folder
    load_hvac_documents_from_folder("hvac_documents/")
