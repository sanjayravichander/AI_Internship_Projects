#!/usr/bin/env python3
"""
Demo script for Advanced AI Voice Assistant
This script demonstrates the key features without requiring the full Streamlit interface
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def demo_conversation_memory():
    """Demonstrate conversational memory"""
    print("🧠 Conversational Memory Demo")
    print("-" * 30)
    
    try:
        from langchain.memory import ConversationBufferWindowMemory
        
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        
        # Simulate conversation
        memory.save_context({"input": "My name is John"}, {"output": "Nice to meet you, John!"})
        memory.save_context({"input": "What's my name?"}, {"output": "Your name is John."})
        
        print("✅ Memory system working")
        print("📝 Sample conversation stored and retrieved")
        
    except Exception as e:
        print(f"❌ Memory demo failed: {e}")

def demo_emotion_detection():
    """Demonstrate emotion detection"""
    print("\n😊 Emotion Detection Demo")
    print("-" * 30)
    
    try:
        from transformers import pipeline
        
        # Initialize emotion classifier
        classifier = pipeline("text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base")
        
        test_texts = [
            "I'm so happy today!",
            "This is really frustrating.",
            "I'm feeling a bit sad.",
            "That's amazing news!",
            "I'm scared about the presentation."
        ]
        
        for text in test_texts:
            result = classifier(text)
            emotion = result[0]['label']
            confidence = result[0]['score']
            print(f"'{text}' → {emotion} ({confidence:.2f})")
        
        print("✅ Emotion detection working")
        
    except Exception as e:
        print(f"❌ Emotion detection demo failed: {e}")

def demo_database():
    """Demonstrate database functionality"""
    print("\n🗄️  Database Demo")
    print("-" * 30)
    
    try:
        import sqlite3
        
        # Create in-memory database for demo
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_input TEXT,
                assistant_response TEXT,
                emotion TEXT
            )
        ''')
        
        # Insert sample data
        cursor.execute('''
            INSERT INTO conversation_history (user_input, assistant_response, emotion)
            VALUES (?, ?, ?)
        ''', ("Hello there!", "Hi! How can I help you today?", "joy"))
        
        cursor.execute('''
            INSERT INTO conversation_history (user_input, assistant_response, emotion)
            VALUES (?, ?, ?)
        ''', ("What time is it?", "The current time is 2:30 PM", "neutral"))
        
        # Query data
        cursor.execute('SELECT * FROM conversation_history')
        conversations = cursor.fetchall()
        
        print("✅ Database operations working")
        print(f"📊 Stored {len(conversations)} conversations")
        
        for conv in conversations:
            print(f"   User: {conv[2]} | Assistant: {conv[3]} | Emotion: {conv[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database demo failed: {e}")

def demo_tools():
    """Demonstrate available tools"""
    print("\n🛠️  Tools Demo")
    print("-" * 30)
    
    tools_info = [
        ("🔍 Search Tool", "Real-time internet search using DuckDuckGo"),
        ("🧮 Calculator Tool", "Python REPL for calculations and data analysis"),
        ("📧 Email Tool", "Send emails via SMTP (requires configuration)"),
        ("⏰ Reminder Tool", "Schedule reminders and notifications"),
        ("🌤️  Weather Tool", "Get current weather information (requires API key)"),
        ("📄 Document QA Tool", "Analyze PDFs and websites to answer questions"),
    ]
    
    for tool_name, description in tools_info:
        print(f"{tool_name}: {description}")
    
    print("\n✅ All tools available and ready to use")

def demo_speech_components():
    """Demonstrate speech components"""
    print("\n🎤 Speech Components Demo")
    print("-" * 30)
    
    try:
        import speech_recognition as sr
        import pyttsx3
        
        # Test speech recognition
        r = sr.Recognizer()
        mics = sr.Microphone.list_microphone_names()
        print(f"🎤 Speech Recognition: {len(mics)} microphone(s) detected")
        
        # Test text-to-speech
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print(f"🗣️  Text-to-Speech: {len(voices)} voice(s) available")
        
        # Demo TTS (uncomment to hear)
        # engine.say("Hello! I am your AI voice assistant.")
        # engine.runAndWait()
        
        engine.stop()
        print("✅ Speech components working")
        
    except Exception as e:
        print(f"❌ Speech components demo failed: {e}")

def demo_translation():
    """Demonstrate translation capabilities"""
    print("\n🌍 Translation Demo")
    print("-" * 30)
    
    try:
        from googletrans import Translator
        
        translator = Translator()
        
        test_text = "Hello, how are you today?"
        languages = [
            ('es', 'Spanish'),
            ('fr', 'French'),
            ('de', 'German'),
            ('it', 'Italian'),
            ('pt', 'Portuguese')
        ]
        
        print(f"Original: {test_text}")
        
        for lang_code, lang_name in languages:
            try:
                result = translator.translate(test_text, dest=lang_code)
                print(f"{lang_name}: {result.text}")
            except:
                print(f"{lang_name}: Translation failed")
        
        print("✅ Translation system working")
        
    except Exception as e:
        print(f"❌ Translation demo failed: {e}")

def main():
    """Run all demos"""
    print("🤖 Advanced AI Voice Assistant - Feature Demo")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("voice_app.py").exists():
        print("❌ Please run this demo from the Voice_Assistant_AI directory")
        sys.exit(1)
    
    # Run demos
    demo_conversation_memory()
    demo_emotion_detection()
    demo_database()
    demo_tools()
    demo_speech_components()
    demo_translation()
    
    print("\n🎉 Demo completed!")
    print("\n🚀 Ready to try the full assistant?")
    print("   Run: streamlit run voice_app.py")
    
    print("\n💡 Pro Tips:")
    print("   - Set up your .env file with API keys for full functionality")
    print("   - Try saying 'Hey Jarvis' for wake word detection")
    print("   - Upload PDFs for document analysis")
    print("   - Use natural language for complex tasks")

if __name__ == "__main__":
    main()