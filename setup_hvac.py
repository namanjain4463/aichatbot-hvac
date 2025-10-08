#!/usr/bin/env python3
"""
Setup script for HVAC Code Chatbot
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages for HVAC Code Chatbot...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("✅ spaCy model downloaded successfully!")

def create_directories():
    """Create necessary directories"""
    directories = ["hvac_documents", "atlanta_codes"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def main():
    print("🏗️ Setting up HVAC Code Chatbot for Atlanta...")
    
    try:
        install_requirements()
        download_spacy_model()
        create_directories()
        
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your HVAC PDF documents to the 'hvac_documents/' folder")
        print("2. Make sure Neo4j is running on localhost:7687")
        print("3. Get your OpenAI API key from https://platform.openai.com/")
        print("4. Run: streamlit run hvac_code_chatbot.py")
        print("\nThe chatbot will automatically load PDFs from the hvac_documents/ folder!")
        
    except Exception as e:
        print(f"❌ Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
