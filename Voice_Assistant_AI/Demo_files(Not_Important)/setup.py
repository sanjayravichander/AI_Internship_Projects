#!/usr/bin/env python3
"""
Setup script for Advanced AI Voice Assistant
This script helps users set up the environment and dependencies
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def setup_environment():
    """Set up environment file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔧 Setting up environment file...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file with your API keys")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example found, creating basic .env file")
        env_content = """# AI Voice Assistant Environment Variables
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
WEATHER_API_KEY=your_weather_api_key_here
"""
        env_file.write_text(env_content)
        print("✅ Basic .env file created")

def setup_database():
    """Initialize the SQLite database"""
    print("🗄️  Setting up database...")
    try:
        conn = sqlite3.connect('assistant_memory.db')
        cursor = conn.cursor()
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY,
                user_name TEXT,
                favorite_topics TEXT,
                preferred_voice INTEGER,
                language TEXT DEFAULT 'en',
                timezone TEXT DEFAULT 'UTC'
            )
        ''')
        
        # Conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                emotion TEXT,
                context TEXT
            )
        ''')
        
        # Reminders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                description TEXT,
                scheduled_time DATETIME,
                completed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")

def check_microphone():
    """Check if microphone is available"""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        mic_list = sr.Microphone.list_microphone_names()
        if mic_list:
            print(f"✅ Microphone detected: {len(mic_list)} device(s) found")
        else:
            print("⚠️  No microphone detected")
    except ImportError:
        print("⚠️  Speech recognition not installed yet")
    except Exception as e:
        print(f"⚠️  Microphone check failed: {e}")

def main():
    """Main setup function"""
    print("🤖 Advanced AI Voice Assistant Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Setup database
    setup_database()
    
    # Check microphone
    check_microphone()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Edit the .env file with your API keys")
    print("2. Run: streamlit run voice_app.py")
    print("3. Start talking to your AI assistant!")
    
    print("\n🔑 Required API Keys:")
    print("- Groq API Key (Required): https://console.groq.com/")
    print("- OpenAI API Key (Optional): https://platform.openai.com/")
    print("- Weather API Key (Optional): https://openweathermap.org/api")

if __name__ == "__main__":
    main()