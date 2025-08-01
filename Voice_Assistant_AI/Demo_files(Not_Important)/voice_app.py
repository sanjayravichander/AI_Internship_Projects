import streamlit as st
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import threading
import sqlite3
import smtplib
import json
import time
import asyncio
import requests
import matplotlib.pyplot as plt
import io
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from transformers import pipeline
import whisper
from googletrans import Translator
from PIL import Image
import numpy as np

# LangChain imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")

# Initialize advanced components
whisper_model = None
emotion_classifier = None
translator = Translator()
scheduler = BackgroundScheduler()
scheduler.start()

# Database setup for user preferences and memory
def init_database():
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

init_database()

# Initialize LangChain + Groq with memory
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.7)

# Conversational Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,  # Remember last 10 exchanges
    return_messages=True
)

# Custom Tools for the Agent
class EmailTool(BaseTool):
    name: str = "email_sender"
    description: str = "Send emails to specified recipients with subject and body"
    
    def _run(self, recipient: str, subject: str, body: str) -> str:
        try:
            if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
                return "Email credentials not configured"
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            return f"Email sent successfully to {recipient}"
        except Exception as e:
            return f"Failed to send email: {str(e)}"

class ReminderTool(BaseTool):
    name: str = "reminder_scheduler"
    description: str = "Schedule reminders for specific dates and times"
    
    def _run(self, title: str, description: str, scheduled_time: str) -> str:
        try:
            # Parse the scheduled time
            reminder_time = datetime.datetime.strptime(scheduled_time, "%Y-%m-%d %H:%M")
            
            # Store in database
            conn = sqlite3.connect('assistant_memory.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reminders (title, description, scheduled_time)
                VALUES (?, ?, ?)
            ''', (title, description, reminder_time))
            conn.commit()
            conn.close()
            
            # Schedule the reminder
            scheduler.add_job(
                func=self._trigger_reminder,
                trigger=DateTrigger(run_date=reminder_time),
                args=[title, description],
                id=f"reminder_{int(time.time())}"
            )
            
            return f"Reminder '{title}' scheduled for {scheduled_time}"
        except Exception as e:
            return f"Failed to schedule reminder: {str(e)}"
    
    def _trigger_reminder(self, title: str, description: str):
        st.warning(f"⏰ Reminder: {title} - {description}")

class WeatherTool(BaseTool):
    name: str = "weather_info"
    description: str = "Get current weather information for any city"
    
    def _run(self, city: str) -> str:
        try:
            # Using a free weather API (you'll need to get an API key)
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                return "Weather API key not configured"
            
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                temp = data['main']['temp']
                description = data['weather'][0]['description']
                return f"Weather in {city}: {temp}°C, {description}"
            else:
                return f"Could not get weather for {city}"
        except Exception as e:
            return f"Weather lookup failed: {str(e)}"

class DocumentQATool(BaseTool):
    name: str = "document_qa"
    description: str = "Answer questions about uploaded documents or websites"
    
    def _run(self, query: str, source: str = "") -> str:
        try:
            if source.startswith("http"):
                loader = WebBaseLoader(source)
            elif source.endswith(".pdf"):
                loader = PyPDFLoader(source)
            else:
                return "Please provide a valid URL or PDF path"
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(texts, embeddings)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            
            result = qa_chain.run(query)
            return result
        except Exception as e:
            return f"Document QA failed: {str(e)}"

# Initialize tools
search_tool = DuckDuckGoSearchRun()
python_tool = PythonREPLTool()
email_tool = EmailTool()
reminder_tool = ReminderTool()
weather_tool = WeatherTool()
doc_qa_tool = DocumentQATool()

tools = [
    Tool(name="Search", func=search_tool.run, description="Search the internet for current information"),
    Tool(name="Calculator", func=python_tool.run, description="Execute Python code for calculations and data analysis"),
    Tool(name="Email", func=email_tool.run, description="Send emails to recipients"),
    Tool(name="Reminder", func=reminder_tool.run, description="Schedule reminders and tasks"),
    Tool(name="Weather", func=weather_tool.run, description="Get weather information for cities"),
    Tool(name="DocumentQA", func=doc_qa_tool.run, description="Answer questions about documents or websites")
]

# Initialize the agent with tools and memory
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# Enhanced Speech components
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Load user preferences
def load_user_preferences():
    conn = sqlite3.connect('assistant_memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_preferences LIMIT 1')
    prefs = cursor.fetchone()
    conn.close()
    return prefs

def save_user_preferences(user_name, favorite_topics, preferred_voice, language):
    conn = sqlite3.connect('assistant_memory.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_preferences 
        (id, user_name, favorite_topics, preferred_voice, language)
        VALUES (1, ?, ?, ?, ?)
    ''', (user_name, favorite_topics, preferred_voice, language))
    conn.commit()
    conn.close()

# Emotion Recognition
def detect_emotion(text):
    global emotion_classifier
    if emotion_classifier is None:
        try:
            emotion_classifier = pipeline("text-classification", 
                                        model="j-hartmann/emotion-english-distilroberta-base")
        except:
            return "neutral"
    
    try:
        result = emotion_classifier(text)
        return result[0]['label'].lower()
    except:
        return "neutral"

# Enhanced TTS with emotion
def speak(text, emotion="neutral"):
    def _speak():
        # Adjust voice properties based on emotion
        if emotion in ["joy", "happy"]:
            tts_engine.setProperty('rate', 200)
            tts_engine.setProperty('volume', 0.9)
        elif emotion in ["sad", "fear"]:
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.7)
        elif emotion in ["anger"]:
            tts_engine.setProperty('rate', 180)
            tts_engine.setProperty('volume', 1.0)
        else:
            tts_engine.setProperty('rate', 175)
            tts_engine.setProperty('volume', 0.8)
        
        tts_engine.say(text)
        tts_engine.runAndWait()

    st.session_state["last_response"] = text
    threading.Thread(target=_speak).start()

# Enhanced listening with Whisper option
def listen(use_whisper=False):
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            if use_whisper and whisper_model:
                # Save audio to temporary file for Whisper
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                result = whisper_model.transcribe("temp_audio.wav")
                os.remove("temp_audio.wav")
                return result["text"].lower()
            else:
                command = recognizer.recognize_google(audio).lower()
                return command
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition error"
        except Exception as e:
            return f"Recognition error: {str(e)}"

# Background listening for wake word
class BackgroundListener:
    def __init__(self, wake_word="hey jarvis"):
        self.wake_word = wake_word.lower()
        self.listening = False
        self.thread = None
    
    def start_listening(self):
        if not self.listening:
            self.listening = True
            self.thread = threading.Thread(target=self._listen_continuously)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_listening(self):
        self.listening = False
    
    def _listen_continuously(self):
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
        
        while self.listening:
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio).lower()
                    
                    if self.wake_word in text:
                        st.session_state["wake_word_detected"] = True
                        st.rerun()
            except:
                continue

# Translation support
def translate_text(text, target_language='en'):
    try:
        result = translator.translate(text, dest=target_language)
        return result.text
    except:
        return text

# Save conversation to memory
def save_conversation(user_input, assistant_response, emotion, context=""):
    conn = sqlite3.connect('assistant_memory.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversation_history 
        (user_input, assistant_response, emotion, context)
        VALUES (?, ?, ?, ?)
    ''', (user_input, assistant_response, emotion, context))
    conn.commit()
    conn.close()

# Enhanced command handler with agent
def handle_command(command, user_prefs=None):
    try:
        # Detect emotion
        emotion = detect_emotion(command)
        
        # Translate if needed
        if user_prefs and user_prefs[4] != 'en':  # language preference
            command = translate_text(command, 'en')
        
        # Use the agent for complex reasoning
        response = agent.run(command)
        
        # Translate response back if needed
        if user_prefs and user_prefs[4] != 'en':
            response = translate_text(response, user_prefs[4])
        
        # Save to conversation history
        save_conversation(command, response, emotion)
        
        return response, emotion
        
    except Exception as e:
        error_msg = f"I encountered an error: {str(e)}"
        save_conversation(command, error_msg, "neutral", "error")
        return error_msg, "neutral"

# Streamlit UI
st.set_page_config(page_title="Advanced AI Voice Assistant", layout="wide")
st.title("🤖 Advanced AI Voice Assistant with Agentic Capabilities")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # User preferences
    user_name = st.text_input("👤 Your Name:", value="User")
    assistant_name = st.text_input("🤖 Assistant Name:", value="Jarvis")
    
    # Voice settings
    voices = tts_engine.getProperty('voices')
    voice_options = [f"Voice {i}: {voice.name}" for i, voice in enumerate(voices)]
    selected_voice = st.selectbox("🗣️ Voice:", voice_options)
    voice_index = int(selected_voice.split(":")[0].split()[1])
    tts_engine.setProperty('voice', voices[voice_index].id)
    
    # Language settings
    languages = {
        'English': 'en', 'Spanish': 'es', 'French': 'fr', 
        'German': 'de', 'Italian': 'it', 'Portuguese': 'pt'
    }
    selected_language = st.selectbox("🌍 Language:", list(languages.keys()))
    language_code = languages[selected_language]
    
    # Advanced options
    st.subheader("🔧 Advanced Options")
    use_whisper = st.checkbox("Use Whisper STT (More Accurate)")
    enable_background_listening = st.checkbox("Background Wake Word Detection")
    multimodal_mode = st.checkbox("Enable Multimodal (Text + Voice)")
    
    # Save preferences
    if st.button("💾 Save Preferences"):
        save_user_preferences(user_name, "", voice_index, language_code)
        st.success("Preferences saved!")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "background_listener" not in st.session_state:
    st.session_state["background_listener"] = BackgroundListener(f"hey {assistant_name.lower()}")
if "wake_word_detected" not in st.session_state:
    st.session_state["wake_word_detected"] = False

# Load Whisper model if requested
if use_whisper and whisper_model is None:
    with st.spinner("Loading Whisper model..."):
        try:
            whisper_model = whisper.load_model("base")
            st.success("Whisper model loaded!")
        except:
            st.error("Failed to load Whisper model")

# Background listening control
if enable_background_listening:
    if not st.session_state["background_listener"].listening:
        st.session_state["background_listener"].start_listening()
        st.info(f"🎧 Background listening active. Say 'Hey {assistant_name}' to activate.")
else:
    if st.session_state["background_listener"].listening:
        st.session_state["background_listener"].stop_listening()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    # Voice input
    if st.button("🎤 Start Voice Command", type="primary") or st.session_state.get("wake_word_detected", False):
        if st.session_state.get("wake_word_detected", False):
            st.session_state["wake_word_detected"] = False
            st.info("Wake word detected! Listening for command...")
        
        command = listen(use_whisper)
        if command and "error" not in command.lower():
            user_prefs = load_user_preferences()
            response, emotion = handle_command(command, user_prefs)
            
            # Add to conversation history
            st.session_state["conversation_history"].append({
                "user": command,
                "assistant": response,
                "emotion": emotion,
                "timestamp": datetime.datetime.now()
            })
            
            speak(response, emotion)
    
    # Text input for multimodal
    if multimodal_mode:
        text_command = st.text_input("💬 Or type your command:")
        if st.button("📝 Send Text Command") and text_command:
            user_prefs = load_user_preferences()
            response, emotion = handle_command(text_command, user_prefs)
            
            st.session_state["conversation_history"].append({
                "user": text_command,
                "assistant": response,
                "emotion": emotion,
                "timestamp": datetime.datetime.now()
            })
            
            speak(response, emotion)

with col2:
    st.subheader("📊 Quick Stats")
    
    # Show conversation count
    conn = sqlite3.connect('assistant_memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conversation_history')
    conv_count = cursor.fetchone()[0]
    st.metric("Total Conversations", conv_count)
    
    # Show reminders count
    cursor.execute('SELECT COUNT(*) FROM reminders WHERE completed = FALSE')
    reminder_count = cursor.fetchone()[0]
    st.metric("Active Reminders", reminder_count)
    conn.close()
    
    # Emotion distribution
    if st.session_state["conversation_history"]:
        emotions = [conv["emotion"] for conv in st.session_state["conversation_history"]]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%')
        ax.set_title("Emotion Distribution")
        st.pyplot(fig)

# Conversation History
st.subheader("💬 Recent Conversations")
for i, conv in enumerate(reversed(st.session_state["conversation_history"][-5:])):
    with st.expander(f"Conversation {len(st.session_state['conversation_history']) - i} - {conv['emotion'].title()} - {conv['timestamp'].strftime('%H:%M')}"):
        st.write(f"**You:** {conv['user']}")
        st.write(f"**{assistant_name}:** {conv['assistant']}")

# File upload for document QA
st.subheader("📄 Document Analysis")
uploaded_file = st.file_uploader("Upload a PDF or enter a URL for analysis", type=['pdf'])
if uploaded_file:
    # Save uploaded file temporarily
    with open("temp_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    doc_question = st.text_input("Ask a question about the document:")
    if st.button("🔍 Analyze Document") and doc_question:
        with st.spinner("Analyzing document..."):
            result = doc_qa_tool.run(f"{doc_question}|temp_document.pdf")
            st.write(result)
        os.remove("temp_document.pdf")

# Quick Actions
st.subheader("⚡ Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🕐 What time is it?"):
        now = datetime.datetime.now()
        response = f"The current time is {now.strftime('%I:%M %p on %B %d, %Y')}"
        speak(response)
        st.info(response)

with col2:
    if st.button("🌤️ Weather"):
        city = st.text_input("Enter city name:", key="weather_city")
        if city:
            weather_response = weather_tool.run(city)
            speak(weather_response)
            st.info(weather_response)

with col3:
    if st.button("📧 Quick Email"):
        with st.form("email_form"):
            recipient = st.text_input("To:")
            subject = st.text_input("Subject:")
            body = st.text_area("Message:")
            if st.form_submit_button("Send"):
                result = email_tool.run(f"{recipient}|{subject}|{body}")
                st.info(result)

with col4:
    if st.button("⏰ Set Reminder"):
        with st.form("reminder_form"):
            title = st.text_input("Reminder Title:")
            description = st.text_area("Description:")
            date = st.date_input("Date:")
            time_input = st.time_input("Time:")
            if st.form_submit_button("Schedule"):
                scheduled_datetime = datetime.datetime.combine(date, time_input)
                result = reminder_tool.run(f"{title}|{description}|{scheduled_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.info(result)

# Footer
st.markdown("---")
st.markdown("🚀 **Advanced Features Active:** Conversational Memory | Agentic Tools | Natural Language Tasks | Emotion Recognition | Multimodal | Background Listening | Translation Support")
