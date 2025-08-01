#!/usr/bin/env python3
"""
Test script to verify the AI Voice Assistant installation
Run this script to check if all components are working correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('speech_recognition', 'Speech recognition'),
        ('pyttsx3', 'Text-to-speech'),
        ('langchain', 'LangChain framework'),
        ('langchain_groq', 'Groq integration'),
        ('transformers', 'Hugging Face transformers'),
        ('googletrans', 'Google Translate'),
        ('apscheduler', 'Task scheduler'),
        ('sqlite3', 'SQLite database'),
        ('requests', 'HTTP requests'),
        ('matplotlib', 'Plotting library'),
        ('numpy', 'Numerical computing'),
        ('pandas', 'Data manipulation'),
    ]
    
    failed_imports = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError as e:
            print(f"  ❌ {package} - {description} (Error: {e})")
            failed_imports.append(package)
    
    return failed_imports

def test_optional_imports():
    """Test optional packages"""
    print("\n🔧 Testing optional imports...")
    
    optional_packages = [
        ('whisper', 'OpenAI Whisper (for advanced STT)'),
        ('faiss', 'FAISS vector database'),
        ('sentence_transformers', 'Sentence transformers'),
    ]
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️  {package} - {description} (Optional - not installed)")

def test_environment():
    """Test environment configuration"""
    print("\n🌍 Testing environment...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("  ✅ .env file found")
        
        # Check for required variables
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'GROQ_API_KEY' in content:
            print("  ✅ GROQ_API_KEY found in .env")
        else:
            print("  ⚠️  GROQ_API_KEY not found in .env")
            
        if 'EMAIL_ADDRESS' in content:
            print("  ✅ EMAIL_ADDRESS found in .env")
        else:
            print("  ⚠️  EMAIL_ADDRESS not found in .env (optional)")
    else:
        print("  ❌ .env file not found")

def test_database():
    """Test database functionality"""
    print("\n🗄️  Testing database...")
    
    try:
        import sqlite3
        conn = sqlite3.connect(':memory:')  # Test with in-memory database
        cursor = conn.cursor()
        
        # Test table creation
        cursor.execute('''
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        ''')
        
        # Test insert
        cursor.execute("INSERT INTO test_table (name) VALUES (?)", ("test",))
        
        # Test select
        cursor.execute("SELECT * FROM test_table")
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            print("  ✅ Database operations working")
        else:
            print("  ❌ Database operations failed")
            
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")

def test_speech_components():
    """Test speech recognition and TTS"""
    print("\n🎤 Testing speech components...")
    
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        print("  ✅ Speech recognition initialized")
        
        # Check for microphones
        mics = sr.Microphone.list_microphone_names()
        if mics:
            print(f"  ✅ {len(mics)} microphone(s) detected")
        else:
            print("  ⚠️  No microphones detected")
            
    except Exception as e:
        print(f"  ❌ Speech recognition test failed: {e}")
    
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if voices:
            print(f"  ✅ Text-to-speech initialized with {len(voices)} voice(s)")
        else:
            print("  ⚠️  No TTS voices found")
        engine.stop()
    except Exception as e:
        print(f"  ❌ Text-to-speech test failed: {e}")

def test_langchain_components():
    """Test LangChain components"""
    print("\n🔗 Testing LangChain components...")
    
    try:
        from langchain.agents import initialize_agent, Tool, AgentType
        from langchain.memory import ConversationBufferWindowMemory
        from langchain_community.tools import DuckDuckGoSearchRun
        print("  ✅ LangChain agent components imported")
    except Exception as e:
        print(f"  ❌ LangChain agent test failed: {e}")
    
    try:
        from langchain_groq import ChatGroq
        print("  ✅ Groq integration imported")
    except Exception as e:
        print(f"  ❌ Groq integration test failed: {e}")

def main():
    """Run all tests"""
    print("🤖 AI Voice Assistant Installation Test")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test optional imports
    test_optional_imports()
    
    # Test environment
    test_environment()
    
    # Test database
    test_database()
    
    # Test speech components
    test_speech_components()
    
    # Test LangChain components
    test_langchain_components()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} required packages failed to import:")
        for package in failed_imports:
            print(f"   - {package}")
        print("\n💡 Run: pip install -r requirements.txt")
    else:
        print("✅ All required packages imported successfully")
    
    if not Path(".env").exists():
        print("⚠️  .env file not found - create one with your API keys")
    
    print("\n🚀 If all tests pass, you can run:")
    print("   streamlit run voice_app.py")

if __name__ == "__main__":
    main()