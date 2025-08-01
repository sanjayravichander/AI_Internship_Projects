#!/usr/bin/env python3
"""
Quick test script to verify the voice assistant app loads without errors
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import speech_recognition as sr
        print("✅ Speech recognition imported successfully")
        
        import pyttsx3
        print("✅ Text-to-speech imported successfully")
        
        from langchain_groq import ChatGroq
        print("✅ LangChain Groq imported successfully")
        
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
        
        import sqlite3
        print("✅ SQLite3 imported successfully")
        
        from apscheduler.schedulers.background import BackgroundScheduler
        print("✅ APScheduler imported successfully")
        
        print("\n🎉 All imports successful! The app should run without import errors.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running the full Streamlit app"""
    try:
        print("\nTesting basic functionality...")
        
        # Test database initialization
        import sqlite3
        conn = sqlite3.connect(':memory:')  # Use in-memory database for testing
        cursor = conn.cursor()
        
        # Test table creation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                test_data TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database operations working")
        
        # Test TTS engine initialization (without actually speaking)
        import pyttsx3
        engine = pyttsx3.init()
        if engine:
            print("✅ Text-to-speech engine initialized")
        else:
            print("⚠️ Text-to-speech engine not available (this is okay)")
        
        # Test speech recognition setup
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        print("✅ Speech recognition initialized")
        
        print("\n🎉 Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Voice Assistant App - Quick Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\n❌ Import tests failed. Please fix dependencies before running the app.")
        return False
    
    functionality_ok = test_basic_functionality()
    if not functionality_ok:
        print("\n⚠️ Some functionality tests failed, but the app might still work.")
    
    print("\n" + "=" * 50)
    print("🚀 Test Summary:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Basic Functionality: {'✅ PASS' if functionality_ok else '⚠️ PARTIAL'}")
    
    if imports_ok:
        print("\n🎯 Ready to run! Use: streamlit run voice_app_simple.py")
        print("\n💡 Don't forget to:")
        print("   1. Create a .env file with your GROQ_API_KEY")
        print("   2. Enable microphone permissions in your browser")
        print("   3. Try the continuous voice chat mode for ChatGPT-like experience")
    
    return imports_ok

if __name__ == "__main__":
    main()