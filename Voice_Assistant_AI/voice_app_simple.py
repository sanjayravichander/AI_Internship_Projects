import streamlit as st
import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os
import threading
import sqlite3
import json
import time
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger

# LangChain imports
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool

# Load environment
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Initialize scheduler
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
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.7)

# Conversational Memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=10,  # Remember last 10 exchanges
    return_messages=True
)

# Custom Tools for the Agent
class SimpleSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet for current information"
    
    def _run(self, query: str) -> str:
        try:
            # Simple web search simulation (you can integrate with actual search APIs)
            return f"Search results for '{query}': This is a simulated search result. In a real implementation, this would return actual web search results."
        except Exception as e:
            return f"Search failed: {str(e)}"

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform mathematical calculations. Supports +, -, *, /, **, (), sqrt, sin, cos, tan, log"
    
    def _run(self, expression: str) -> str:
        try:
            import math
            import re
            
            # Clean the expression
            expression = expression.strip().lower()
            
            # Replace common math functions
            expression = expression.replace('sqrt', 'math.sqrt')
            expression = expression.replace('sin', 'math.sin')
            expression = expression.replace('cos', 'math.cos')
            expression = expression.replace('tan', 'math.tan')
            expression = expression.replace('log', 'math.log')
            expression = expression.replace('pi', 'math.pi')
            expression = expression.replace('e', 'math.e')
            
            # Safe evaluation - only allow specific characters and functions
            allowed_pattern = r'^[0-9+\-*/.() \s,mathsqrtincosanlogpie]+$'
            if not re.match(allowed_pattern, expression):
                return "Invalid characters in mathematical expression. Only numbers, +, -, *, /, **, (), and basic math functions are allowed."
            
            # Create safe namespace for eval
            safe_dict = {
                "__builtins__": {},
                "math": math,
                "abs": abs,
                "round": round,
                "pow": pow,
                "max": max,
                "min": min
            }
            
            # Evaluate the expression
            result = eval(expression, safe_dict)
            
            # Format the result nicely
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 6)
            
            return f"The result of '{expression.replace('math.', '')}' is {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero is not allowed"
        except ValueError as e:
            return f"Math error: {str(e)}"
        except SyntaxError:
            return "Invalid mathematical expression syntax"
        except Exception as e:
            return f"Calculation error: {str(e)}"

class EmailTool(BaseTool):
    name: str = "email_sender"
    description: str = "Send emails to specified recipients. Format: recipient|subject|body"
    
    def _run(self, input_str: str) -> str:
        try:
            # Parse input: recipient|subject|body
            parts = input_str.split('|')
            if len(parts) < 2:
                return "Please provide at least: recipient|subject (body is optional)"
            elif len(parts) == 2:
                recipient, subject = parts
                body = ""
            else:
                recipient, subject, body = parts[0], parts[1], '|'.join(parts[2:])
            
            # Validate email address format
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, recipient.strip()):
                return f"Invalid email address format: {recipient}"
            
            if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
                return "Email credentials not configured. Please check your .env file has EMAIL_ADDRESS and EMAIL_PASSWORD"
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient.strip()
            msg['Subject'] = subject.strip()
            
            if body:
                msg.attach(MIMEText(body.strip(), 'plain'))
            else:
                msg.attach(MIMEText("(No message body)", 'plain'))
            
            # Send email with better error handling
            try:
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
                server.quit()
                
                return f"✅ Email sent successfully to {recipient.strip()}"
                
            except smtplib.SMTPAuthenticationError:
                return "❌ Email authentication failed. Please check your Gmail App Password in .env file. Regular Gmail password won't work - you need an App Password."
            except smtplib.SMTPRecipientsRefused:
                return f"❌ Recipient email address rejected: {recipient}"
            except smtplib.SMTPServerDisconnected:
                return "❌ SMTP server disconnected. Please try again."
            except Exception as smtp_error:
                return f"❌ SMTP error: {str(smtp_error)}"
                
        except Exception as e:
            return f"❌ Failed to send email: {str(e)}"

class ReminderTool(BaseTool):
    name: str = "reminder_scheduler"
    description: str = "Schedule reminders. Format: title|description|time (e.g., 'Meeting|Team standup|2024-01-15 14:30' or 'Call mom|Weekly call|tomorrow 3pm')"
    
    def _run(self, input_str: str) -> str:
        try:
            # Parse input: title|description|scheduled_time
            parts = input_str.split('|')
            if len(parts) < 2:
                return "Please provide at least: title|time (description is optional)"
            elif len(parts) == 2:
                title, scheduled_time = parts
                description = ""
            else:
                title, description, scheduled_time = parts[0], parts[1], '|'.join(parts[2:])
            
            # Parse the scheduled time with multiple formats
            reminder_time = self._parse_time(scheduled_time.strip())
            if not reminder_time:
                return "Invalid time format. Use formats like: '2024-01-15 14:30', 'tomorrow 3pm', 'in 2 hours', 'next monday 9am'"
            
            # Check if time is in the future
            if reminder_time <= datetime.datetime.now():
                return "Reminder time must be in the future"
            
            # Store in database
            try:
                conn = sqlite3.connect('assistant_memory.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO reminders (title, description, scheduled_time)
                    VALUES (?, ?, ?)
                ''', (title.strip(), description.strip(), reminder_time))
                conn.commit()
                conn.close()
            except Exception as db_error:
                return f"Database error: {str(db_error)}"
            
            # Schedule the reminder
            try:
                job_id = f"reminder_{int(time.time())}_{hash(title) % 10000}"
                scheduler.add_job(
                    func=self._trigger_reminder,
                    trigger=DateTrigger(run_date=reminder_time),
                    args=[title.strip(), description.strip()],
                    id=job_id
                )
                
                formatted_time = reminder_time.strftime("%Y-%m-%d at %I:%M %p")
                return f"✅ Reminder '{title.strip()}' scheduled for {formatted_time}"
                
            except Exception as scheduler_error:
                return f"Scheduler error: {str(scheduler_error)}"
                
        except Exception as e:
            return f"❌ Failed to schedule reminder: {str(e)}"
    
    def _parse_time(self, time_str: str) -> datetime.datetime:
        """Parse various time formats"""
        import re
        from dateutil import parser
        from dateutil.relativedelta import relativedelta
        
        time_str = time_str.lower().strip()
        now = datetime.datetime.now()
        
        try:
            # Try standard formats first
            formats = [
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %I:%M %p",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y %I:%M %p",
                "%d/%m/%Y %H:%M",
                "%Y-%m-%d",
            ]
            
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
            
            # Handle relative times
            if "tomorrow" in time_str:
                base_time = now + datetime.timedelta(days=1)
                time_part = re.search(r'(\d{1,2}):?(\d{0,2})\s*(am|pm)?', time_str)
                if time_part:
                    hour = int(time_part.group(1))
                    minute = int(time_part.group(2)) if time_part.group(2) else 0
                    if time_part.group(3) == 'pm' and hour != 12:
                        hour += 12
                    elif time_part.group(3) == 'am' and hour == 12:
                        hour = 0
                    return base_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
                else:
                    return base_time.replace(hour=9, minute=0, second=0, microsecond=0)
            
            # Handle "in X hours/minutes"
            if "in " in time_str:
                match = re.search(r'in (\d+) (hour|minute|day)s?', time_str)
                if match:
                    amount = int(match.group(1))
                    unit = match.group(2)
                    if unit == 'hour':
                        return now + datetime.timedelta(hours=amount)
                    elif unit == 'minute':
                        return now + datetime.timedelta(minutes=amount)
                    elif unit == 'day':
                        return now + datetime.timedelta(days=amount)
            
            # Try dateutil parser as fallback
            return parser.parse(time_str)
            
        except Exception:
            return None
    
    def _trigger_reminder(self, title: str, description: str):
        """Trigger reminder notification"""
        print(f"⏰ REMINDER: {title}")
        if description:
            print(f"   📝 {description}")
        
        # Try to show in Streamlit if available
        try:
            st.warning(f"⏰ Reminder: {title}" + (f" - {description}" if description else ""))
            st.balloons()
        except:
            pass  # Streamlit not available in background thread

class WeatherTool(BaseTool):
    name: str = "weather_info"
    description: str = "Get current weather information for any city worldwide"
    
    def _run(self, city: str) -> str:
        try:
            api_key = WEATHER_API_KEY
            if not api_key:
                return f"❌ Weather API key not configured. Please add WEATHER_API_KEY to your .env file to get weather for {city}."
            
            # Clean city name
            city = city.strip()
            
            # OpenWeatherMap API call
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            
            try:
                response = requests.get(url, timeout=10)
                data = response.json()
                
                if response.status_code == 200:
                    # Extract weather data
                    temp = round(data['main']['temp'], 1)
                    feels_like = round(data['main']['feels_like'], 1)
                    humidity = data['main']['humidity']
                    description = data['weather'][0]['description'].title()
                    wind_speed = data['wind']['speed']
                    
                    # Format response
                    weather_info = f"🌤️ Weather in {city}:\n"
                    weather_info += f"🌡️ Temperature: {temp}°C (feels like {feels_like}°C)\n"
                    weather_info += f"☁️ Conditions: {description}\n"
                    weather_info += f"💨 Wind: {wind_speed} m/s\n"
                    weather_info += f"💧 Humidity: {humidity}%"
                    
                    return weather_info
                    
                elif response.status_code == 404:
                    return f"❌ City '{city}' not found. Please check the spelling and try again."
                elif response.status_code == 401:
                    return f"❌ Invalid weather API key. Please check your WEATHER_API_KEY in .env file."
                else:
                    return f"❌ Weather service error (code {response.status_code}): {data.get('message', 'Unknown error')}"
                    
            except requests.exceptions.Timeout:
                return f"❌ Weather service timeout. Please try again later."
            except requests.exceptions.ConnectionError:
                return f"❌ Cannot connect to weather service. Please check your internet connection."
            except requests.exceptions.RequestException as req_error:
                return f"❌ Weather request failed: {str(req_error)}"
                
        except Exception as e:
            return f"❌ Weather lookup failed: {str(e)}"

# Initialize tools
search_tool = SimpleSearchTool()
calculator_tool = CalculatorTool()
email_tool = EmailTool()
reminder_tool = ReminderTool()
weather_tool = WeatherTool()

tools = [
    Tool(name="Search", func=search_tool.run, description="Search the internet for current information"),
    Tool(name="Calculator", func=calculator_tool.run, description="Perform mathematical calculations"),
    Tool(name="Email", func=email_tool.run, description="Send emails to recipients (format: recipient|subject|body)"),
    Tool(name="Reminder", func=reminder_tool.run, description="Schedule reminders (format: title|description|YYYY-MM-DD HH:MM)"),
    Tool(name="Weather", func=weather_tool.run, description="Get weather information for cities")
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

# Initialize TTS engine with multiple fallback options
def init_tts_engine():
    """Initialize TTS engine with proper COM handling for Windows"""
    
    # Method 1: Try with COM initialization
    try:
        import pythoncom
        pythoncom.CoInitialize()  # Initialize COM
        engine = pyttsx3.init()
        print("✅ TTS initialized with COM")
        return engine
    except Exception as e:
        print(f"COM initialization failed: {e}")
    
    # Method 2: Try without COM initialization
    try:
        engine = pyttsx3.init()
        print("✅ TTS initialized without COM")
        return engine
    except Exception as e:
        print(f"Direct TTS initialization failed: {e}")
    
    # Method 3: Try with different driver
    try:
        engine = pyttsx3.init(driverName='espeak')
        print("✅ TTS initialized with espeak driver")
        return engine
    except Exception as e:
        print(f"Espeak TTS initialization failed: {e}")
    
    # Method 4: Try with dummy driver (no actual speech)
    print("⚠️ TTS not available - using text-only mode")
    return None

# Alternative TTS using Windows SAPI directly
def windows_speak(text):
    """Fallback TTS using Windows SAPI via subprocess"""
    try:
        import subprocess
        # Use Windows built-in SAPI via PowerShell
        cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{text}\')"'
        subprocess.run(cmd, shell=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Windows SAPI fallback failed: {e}")
        return False

tts_engine = init_tts_engine()

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

# Simple emotion detection based on keywords
def detect_emotion(text):
    text = text.lower()
    if any(word in text for word in ['happy', 'great', 'awesome', 'wonderful', 'excited']):
        return 'joy'
    elif any(word in text for word in ['sad', 'upset', 'disappointed', 'down']):
        return 'sad'
    elif any(word in text for word in ['angry', 'mad', 'frustrated', 'annoyed']):
        return 'anger'
    elif any(word in text for word in ['scared', 'afraid', 'worried', 'nervous']):
        return 'fear'
    else:
        return 'neutral'

# Enhanced TTS with emotion and better reliability
def speak(text, emotion="neutral"):
    """Speak text with emotion and multiple fallback options"""
    
    def _speak():
        success = False
        
        # Method 1: Try pyttsx3 first
        if tts_engine is not None:
            try:
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
                
                # Clean text for better speech
                clean_text = text.replace("*", "").replace("**", "").strip()
                
                tts_engine.say(clean_text)
                tts_engine.runAndWait()
                success = True
                print(f"✅ TTS: Spoke via pyttsx3")
                
            except Exception as e:
                print(f"⚠️ pyttsx3 TTS error: {e}")
        
        # Method 2: Fallback to Windows SAPI
        if not success:
            try:
                clean_text = text.replace("*", "").replace("**", "").strip()
                if windows_speak(clean_text):
                    success = True
                    print(f"✅ TTS: Spoke via Windows SAPI")
            except Exception as e:
                print(f"⚠️ Windows SAPI error: {e}")
        
        # Method 3: Final fallback - display text
        if not success:
            print(f"🔊 TTS (Text-only): {text}")

    # Store response and start speaking
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = ""
    
    st.session_state["last_response"] = text
    
    # Start TTS in separate thread to avoid blocking
    try:
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    except Exception as e:
        print(f"⚠️ TTS threading error: {e}")
        print(f"🔊 TTS (Direct): {text}")

# Enhanced listening with better sentence handling
def listen():
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... (Speak now)")
            
            # Adjust for ambient noise with shorter duration
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Shorter, more responsive timeouts
            # timeout=3: Wait up to 3 seconds for speech to start
            # phrase_time_limit=8: Allow up to 8 seconds for complete sentence
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=8)
            
            st.info("🔄 Processing your speech...")
            
            # Use Google Speech Recognition
            command = recognizer.recognize_google(audio)
            st.success(f"✅ Heard: {command}")
            return command.lower()
            
    except sr.WaitTimeoutError:
        return "timeout_error"
    except sr.UnknownValueError:
        return "unclear_audio"
    except sr.RequestError as e:
        return f"service_error: {str(e)}"
    except Exception as e:
        return f"recognition_error: {str(e)}"

# Background listening for wake word
class BackgroundListener:
    def __init__(self, wake_word="hey jarvis"):
        self.wake_word = wake_word.lower()
        self.listening = False
        self.thread = None
        self.wake_detected = False
    
    def start_listening(self):
        if not self.listening:
            self.listening = True
            self.wake_detected = False
            self.thread = threading.Thread(target=self._listen_continuously)
            self.thread.daemon = True
            self.thread.start()
    
    def stop_listening(self):
        self.listening = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
    
    def check_wake_word(self):
        """Check if wake word was detected and reset the flag"""
        if self.wake_detected:
            self.wake_detected = False
            return True
        return False
    
    def update_wake_word(self, new_wake_word):
        """Update the wake word"""
        self.wake_word = new_wake_word.lower()
    
    def _listen_continuously(self):
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
        except:
            return
        
        while self.listening:
            try:
                with sr.Microphone() as source:
                    # Shorter timeout to prevent blocking
                    audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
                    text = recognizer.recognize_google(audio).lower()
                    
                    if self.wake_word in text:
                        self.wake_detected = True
                        print(f"Wake word detected: {text}")  # Debug print
                        
            except sr.WaitTimeoutError:
                # This is expected, just continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand audio, continue
                continue
            except sr.RequestError:
                # API error, continue
                continue
            except Exception as e:
                # Other errors, continue but print for debugging
                print(f"Background listener error: {e}")
                continue

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

# Application launcher function
def launch_application(command):
    """Launch applications based on voice command"""
    import subprocess
    import shutil
    
    # Define application mappings (app name -> executable/command)
    app_mappings = {
        # Browsers
        'chrome': ['chrome', 'google chrome'],
        'firefox': ['firefox', 'mozilla firefox'],
        'edge': ['msedge', 'microsoft edge', 'edge'],
        'opera': ['opera'],
        
        # Microsoft Office
        'word': ['winword', 'microsoft word', 'ms word'],
        'excel': ['excel', 'microsoft excel', 'ms excel'],
        'powerpoint': ['powerpnt', 'microsoft powerpoint', 'ms powerpoint', 'ppt'],
        'outlook': ['outlook', 'microsoft outlook', 'ms outlook'],
        'onenote': ['onenote', 'microsoft onenote', 'ms onenote'],
        
        # Media & Entertainment
        'spotify': ['spotify'],
        'vlc': ['vlc', 'vlc media player'],
        'wmplayer': ['wmplayer', 'windows media player', 'media player'],
        'photos': ['ms-photos:', 'windows photos'],
        
        # Development Tools
        'visual studio code': ['code', 'vscode', 'vs code'],
        'visual studio': ['devenv', 'visual studio'],
        'notepad': ['notepad'],
        'notepad++': ['notepad++', 'notepadplusplus'],
        'cmd': ['cmd', 'command prompt', 'terminal'],
        'powershell': ['powershell', 'windows powershell'],
        
        # System Apps
        'calculator': ['calc', 'calculator'],
        'paint': ['mspaint', 'ms paint', 'microsoft paint'],
        'file explorer': ['explorer', 'windows explorer', 'file manager'],
        'task manager': ['taskmgr', 'task manager'],
        'control panel': ['control', 'control panel'],
        'settings': ['ms-settings:', 'windows settings', 'pc settings'],
        
        # Communication
        'teams': ['teams', 'microsoft teams', 'ms teams'],
        'skype': ['skype'],
        'discord': ['discord'],
        'zoom': ['zoom'],
        'whatsapp': ['WhatsApp'],
        'telegram': ['telegram'],
        
        # Other Popular Apps
        'steam': ['steam'],
        'adobe photoshop': ['photoshop'],
        'adobe reader': ['acrord32', 'adobe reader', 'pdf reader'],
        'winrar': ['winrar'],
        '7zip': ['7zfm', '7-zip'],
        
        # Social Media & Web Apps (opens in browser)
        'instagram': ['instagram', 'insta'],
        'facebook': ['facebook', 'fb'],
        'twitter': ['twitter', 'x'],
        'linkedin': ['linkedin'],
        'tiktok': ['tiktok', 'tik tok'],
        'reddit': ['reddit'],
        'pinterest': ['pinterest'],
        'snapchat': ['snapchat'],
        'gmail': ['gmail', 'google mail'],
        'google': ['google', 'google search'],
    }
    
    # Extract app name from command
    command = command.lower().strip()
    
    # Remove trigger words
    for trigger in ['open', 'launch', 'start', 'run']:
        command = command.replace(trigger, '').strip()
    
    # Find matching application
    matched_app = None
    executable = None
    
    for exe, aliases in app_mappings.items():
        for alias in aliases:
            if alias in command:
                matched_app = alias
                executable = exe
                break
        if matched_app:
            break
    
    if not matched_app:
        return None
    
    try:
        # Define web app URLs
        web_apps = {
            'instagram': 'https://www.instagram.com',
            'facebook': 'https://www.facebook.com',
            'twitter': 'https://www.twitter.com',
            'linkedin': 'https://www.linkedin.com',
            'tiktok': 'https://www.tiktok.com',
            'reddit': 'https://www.reddit.com',
            'pinterest': 'https://www.pinterest.com',
            'snapchat': 'https://web.snapchat.com',
            'gmail': 'https://mail.google.com',
            'google': 'https://www.google.com'
        }
        
        # Special handling for different types of applications
        if executable in web_apps:
            # Web applications - open in browser
            webbrowser.open(web_apps[executable])
        elif executable.startswith('ms-'):
            # Windows Store apps or special protocols
            subprocess.run(['start', executable], shell=True, check=True)
        elif executable in ['explorer', 'calc', 'mspaint', 'notepad', 'taskmgr', 'control']:
            # Built-in Windows applications
            subprocess.run([executable], shell=True)
        elif executable == 'cmd':
            subprocess.run(['start', 'cmd'], shell=True)
        elif executable == 'powershell':
            subprocess.run(['start', 'powershell'], shell=True)
        else:
            # Try to find the executable in PATH or common locations
            if shutil.which(executable):
                subprocess.run([executable], shell=True)
            else:
                # Try with start command for Windows
                subprocess.run(['start', '', executable], shell=True)
        
        response = f"✅ Successfully launched {matched_app.title()}!"
        return True, response
        
    except subprocess.CalledProcessError:
        response = f"❌ Failed to launch {matched_app.title()}. The application might not be installed or accessible."
        return True, response
    except FileNotFoundError:
        response = f"❌ {matched_app.title()} not found. Please make sure it's installed on your system."
        return True, response
    except Exception as e:
        response = f"❌ Error launching {matched_app.title()}: {str(e)}"
        return True, response

# Enhanced command handler with agent
def handle_command(command, user_prefs=None, assistant_name="Jarvis"):
    try:
        # Detect emotion
        emotion = detect_emotion(command)
        
        # Convert command to lowercase for easier matching
        command_lower = command.lower().strip()
        
        # Handle specific commands first for better reliability and speed
        
        # Time command handler
        if any(phrase in command_lower for phrase in ["what's the time", "what is the time", "current time", "tell me the time", "what time is it"]):
            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%I:%M %p")
            date_str = current_time.strftime("%A, %B %d, %Y")
            response = f"The current time is {time_str} on {date_str}."
            save_conversation(command, response, emotion)
            return response, emotion
        
        # YouTube command handler
        elif any(phrase in command_lower for phrase in ["open youtube", "go to youtube", "launch youtube", "start youtube"]):
            try:
                webbrowser.open("https://www.youtube.com")
                response = f"Opening YouTube for you! The YouTube website should open in your default browser."
                save_conversation(command, response, emotion)
                return response, emotion
            except Exception as browser_error:
                response = f"I tried to open YouTube, but encountered an issue: {str(browser_error)}. Please try opening your browser manually."
                save_conversation(command, response, emotion, "browser_error")
                return response, emotion
        
        # YouTube search command handler
        elif any(phrase in command_lower for phrase in ["play", "search", "find"]) and "youtube" in command_lower:
            try:
                # Extract search query from the command
                search_query = command_lower
                
                # Remove common trigger words to get the actual search term
                for phrase in ["play", "search for", "find", "on youtube", "in youtube", "youtube"]:
                    search_query = search_query.replace(phrase, "")
                
                # Clean up the search query
                search_query = search_query.strip()
                
                if search_query:
                    # Create YouTube search URL
                    import urllib.parse
                    encoded_query = urllib.parse.quote_plus(search_query)
                    youtube_search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
                    
                    webbrowser.open(youtube_search_url)
                    response = f"Opening YouTube search results for '{search_query}'. You should see the search results in your browser!"
                    save_conversation(command, response, emotion)
                    return response, emotion
                else:
                    webbrowser.open("https://www.youtube.com")
                    response = f"Opening YouTube for you! Please specify what you'd like to search for next time."
                    save_conversation(command, response, emotion)
                    return response, emotion
                    
            except Exception as browser_error:
                response = f"I tried to search YouTube, but encountered an issue: {str(browser_error)}. Please try opening your browser manually."
                save_conversation(command, response, emotion, "browser_error")
                return response, emotion
        
        # Application launcher handler
        elif any(phrase in command_lower for phrase in ["open", "launch", "start", "run"]):
            try:
                app_result = launch_application(command_lower)
                if app_result is not None:
                    app_launched, app_response = app_result
                    if app_launched:
                        save_conversation(command, app_response, emotion)
                        return app_response, emotion
                # If app not found, continue to LLM agent
            except Exception as app_error:
                # If there's an error in app launching, continue to LLM agent
                pass
        
        # For all other commands, use the LLM agent
        else:
            # Create a personalized prompt that includes the assistant name
            personalized_command = f"You are {assistant_name}, a helpful AI assistant. Please respond to: {command}"
            
            # Use the agent for complex reasoning
            response = agent.run(personalized_command)
            
            # Replace any generic references with the assistant name
            response = response.replace("Assistant", assistant_name)
            response = response.replace("assistant", assistant_name)
            response = response.replace("I am an AI", f"I am {assistant_name}, your AI")
            
            # Save to conversation history
            save_conversation(command, response, emotion)
            
            return response, emotion
        
    except Exception as e:
        error_msg = f"I'm {assistant_name}, and I encountered an error: {str(e)}"
        save_conversation(command, error_msg, "neutral", "error")
        return error_msg, "neutral"

# Greeting function
def greet_user(user_name, assistant_name):
    """Generate a personalized greeting"""
    import datetime
    
    current_hour = datetime.datetime.now().hour
    
    if 5 <= current_hour < 12:
        time_greeting = "Good morning"
    elif 12 <= current_hour < 17:
        time_greeting = "Good afternoon"
    elif 17 <= current_hour < 21:
        time_greeting = "Good evening"
    else:
        time_greeting = "Good night"
    
    greeting = f"{time_greeting}, {user_name}! I'm {assistant_name}, your AI voice assistant. How can I help you today?"
    return greeting

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
    if tts_engine is not None:
        try:
            voices = tts_engine.getProperty('voices')
            if voices:
                voice_options = [f"Voice {i}: {voice.name}" for i, voice in enumerate(voices)]
                selected_voice = st.selectbox("🗣️ Voice:", voice_options)
                voice_index = int(selected_voice.split(":")[0].split()[1])
                tts_engine.setProperty('voice', voices[voice_index].id)
            else:
                st.warning("⚠️ No voices available for text-to-speech")
        except Exception as e:
            st.warning(f"⚠️ Voice settings unavailable: {str(e)}")
    else:
        st.warning("⚠️ Text-to-speech engine not available")
    

    
    # Voice Chat Options
    st.subheader("🎤 Voice Chat Options")
    continuous_chat_mode = st.checkbox("Continuous Voice Chat Mode (ChatGPT-like)", value=st.session_state.get("continuous_chat_mode", False))
    voice_response_mode = st.checkbox("Voice Responses", value=st.session_state.get("voice_response_mode", True))
    
    # Update session state
    st.session_state["continuous_chat_mode"] = continuous_chat_mode
    st.session_state["voice_response_mode"] = voice_response_mode
    
    if continuous_chat_mode:
        st.success("🎯 **True Continuous Chat**: Automatically listens after each response - just like ChatGPT!")
        st.info("💡 Say 'stop listening' or click the stop button to end the conversation.")
    
    # Save preferences
    if st.button("💾 Save Preferences"):
        save_user_preferences(user_name, "", 0, 'en')
        st.success("Preferences saved!")

# Get values from session state for use in main interface
continuous_chat_mode = st.session_state.get("continuous_chat_mode", False)
voice_response_mode = st.session_state.get("voice_response_mode", True)

# Initialize session state with better stability
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "background_listener" not in st.session_state:
    st.session_state["background_listener"] = BackgroundListener(f"hey {assistant_name.lower()}")
if "last_assistant_name" not in st.session_state:
    st.session_state["last_assistant_name"] = assistant_name
if "last_user_name" not in st.session_state:
    st.session_state["last_user_name"] = user_name
if "greeted" not in st.session_state:
    st.session_state["greeted"] = False
if "app_initialized" not in st.session_state:
    st.session_state["app_initialized"] = True
if "continuous_chat_mode" not in st.session_state:
    st.session_state["continuous_chat_mode"] = False
if "voice_response_mode" not in st.session_state:
    st.session_state["voice_response_mode"] = True

# Handle user name change and greeting
if st.session_state["last_user_name"] != user_name:
    st.session_state["last_user_name"] = user_name
    st.session_state["greeted"] = False  # Reset greeting when name changes

# Greet user when name is set and not greeted yet
if user_name != "User" and not st.session_state["greeted"]:
    greeting = greet_user(user_name, assistant_name)
    st.success(f"👋 {greeting}")
    speak(greeting, "joy")
    st.session_state["greeted"] = True
    
    # Add greeting to conversation history
    st.session_state["conversation_history"].append({
        "user": f"User set name to: {user_name}",
        "assistant": greeting,
        "emotion": "joy",
        "timestamp": datetime.datetime.now()
    })



# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    # ChatGPT-like Voice Interface
    st.subheader("🎤 Voice Assistant Chat")
    
    # Continuous Chat Mode
    if continuous_chat_mode:
        st.info("🔄 **Continuous Chat Mode Active** - The assistant will keep listening after each response")
        
        # Control buttons for continuous mode
        col_start, col_stop = st.columns([1, 1])
        with col_start:
            start_continuous = st.button("🎤 Start Continuous Chat", type="primary")
        with col_stop:
            stop_continuous = st.button("⏹️ Stop Continuous Chat")
        
        # Initialize continuous chat session state
        if "continuous_active" not in st.session_state:
            st.session_state["continuous_active"] = False
        
        if start_continuous:
            st.session_state["continuous_active"] = True
            st.success("🎤 Continuous chat started! Speak your message...")
        
        if stop_continuous:
            st.session_state["continuous_active"] = False
            st.info("⏹️ Continuous chat stopped.")
        
        # True Continuous Chat Loop (like ChatGPT)
        if st.session_state["continuous_active"]:
            # Display current conversation
            st.subheader("💬 Live Conversation")
            
            # Show recent conversation history
            if st.session_state["conversation_history"]:
                recent_conversations = st.session_state["conversation_history"][-8:]
                for i, conv in enumerate(recent_conversations):
                    st.write(f"**🗣️ You:** {conv['user']}")
                    st.write(f"**🤖 {assistant_name}:** {conv['assistant']}")
                    st.write(f"*{conv['timestamp'].strftime('%H:%M:%S')}*")
                    st.divider()
            
            # Stop button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⏹️ Stop Continuous Chat", type="secondary", key="stop_continuous_main"):
                    st.session_state["continuous_active"] = False
                    st.success("👋 Continuous chat stopped!")
                    st.rerun()
            
            # Automatic continuous listening
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                with status_placeholder.container():
                    st.info("🎤 Listening... (Say 'stop listening' to exit or click the stop button above)")
                
                # Listen for voice input
                command = listen()
                status_placeholder.empty()
                
                if command and len(command.strip()) > 0:
                    # Handle error responses from listen()
                    if command in ["timeout_error", "unclear_audio"] or "service_error" in command or "recognition_error" in command:
                        with response_placeholder.container():
                            if command == "timeout_error":
                                st.warning("⚠️ No speech detected. Listening again...")
                            elif command == "unclear_audio":
                                st.warning("⚠️ Could not understand audio. Please speak clearly.")
                            else:
                                st.warning(f"⚠️ {command}")
                        time.sleep(2)
                        st.rerun()
                    # Check for stop commands
                    elif any(stop_word in command.lower() for stop_word in ["stop listening", "exit chat", "end conversation", "goodbye", "stop continuous"]):
                        st.session_state["continuous_active"] = False
                        with response_placeholder.container():
                            st.success("👋 Continuous chat ended. Goodbye!")
                        if voice_response_mode:
                            speak("Goodbye! Continuous chat ended.", "neutral")
                        time.sleep(2)
                        st.rerun()
                    else:
                        # Process the command
                        with response_placeholder.container():
                            st.write(f"**🗣️ You:** {command}")
                            
                            # Show thinking indicator
                            thinking_placeholder = st.empty()
                            with thinking_placeholder.container():
                                st.info("🤔 Processing your message...")
                            
                            try:
                                user_prefs = load_user_preferences()
                                response, emotion = handle_command(command, user_prefs, assistant_name)
                                
                                thinking_placeholder.empty()
                                st.write(f"**🤖 {assistant_name}:** {response}")
                                
                                # Add to conversation history
                                st.session_state["conversation_history"].append({
                                    "user": command,
                                    "assistant": response,
                                    "emotion": emotion,
                                    "timestamp": datetime.datetime.now()
                                })
                                
                                # Speak response if voice mode is enabled
                                if voice_response_mode:
                                    try:
                                        speak(response, emotion)
                                        st.success("🔊 Response spoken! Listening for your next message...")
                                    except Exception as speak_error:
                                        st.warning(f"⚠️ Voice output error: {speak_error}")
                                else:
                                    st.success("✅ Response ready! Listening for your next message...")
                                
                                # Brief pause before continuing to listen
                                time.sleep(2)
                                st.rerun()
                                
                            except Exception as cmd_error:
                                thinking_placeholder.empty()
                                error_msg = f"Sorry, I encountered an error: {str(cmd_error)}"
                                st.error(error_msg)
                                if voice_response_mode:
                                    try:
                                        speak(error_msg, "neutral")
                                    except:
                                        pass
                                time.sleep(2)
                                st.rerun()
                else:
                    with response_placeholder.container():
                        st.warning("No speech detected. Listening again...")
                    time.sleep(1)
                    st.rerun()
                    
            except Exception as e:
                status_placeholder.empty()
                response_placeholder.empty()
                st.error(f"Voice recognition error: {str(e)}")
                st.info("Restarting continuous listening...")
                time.sleep(2)
                st.rerun()
    
    else:
        # Single Voice Command Mode
        # Voice input button
        voice_button_clicked = st.button("🎤 Voice Message", type="primary")
        
        if voice_button_clicked:
            
            # Create placeholders for dynamic updates
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                with status_placeholder.container():
                    st.info("🎤 Listening for your message...")
                
                command = listen()
                status_placeholder.empty()
                
                if command and len(command.strip()) > 0:
                    # Handle error responses from listen()
                    if command in ["timeout_error", "unclear_audio"] or "service_error" in command or "recognition_error" in command:
                        if command == "timeout_error":
                            st.warning("⚠️ No speech detected. Please try again.")
                        elif command == "unclear_audio":
                            st.warning("⚠️ Could not understand audio. Please speak clearly.")
                        else:
                            st.warning(f"⚠️ {command}")
                    else:
                        # Display user input
                        st.write(f"**🗣️ You:** {command}")
                        
                        with response_placeholder.container():
                            st.info("🤔 Thinking...")
                        
                        try:
                            user_prefs = load_user_preferences()
                            response, emotion = handle_command(command, user_prefs, assistant_name)
                            
                            response_placeholder.empty()
                            st.write(f"**🤖 {assistant_name}:** {response}")
                            
                            # Add to conversation history
                            st.session_state["conversation_history"].append({
                                "user": command,
                                "assistant": response,
                                "emotion": emotion,
                                "timestamp": datetime.datetime.now()
                            })
                            
                            # Speak response if voice mode is enabled
                            if st.session_state["voice_response_mode"]:
                                try:
                                    speak(response, emotion)
                                except Exception as speak_error:
                                    st.warning(f"⚠️ Voice output error: {speak_error}")
                            
                        except Exception as cmd_error:
                            response_placeholder.empty()
                            error_msg = f"Sorry, I encountered an error: {str(cmd_error)}"
                            st.error(error_msg)
                            if st.session_state["voice_response_mode"]:
                                speak(error_msg, "neutral")
                else:
                    st.warning("No speech detected. Please try again.")
                    
            except Exception as e:
                status_placeholder.empty()
                response_placeholder.empty()
                st.error(f"Voice recognition error: {str(e)}")

with col2:
    st.subheader("📊 Quick Stats")
    
    # Show conversation count
    conn = sqlite3.connect('assistant_memory.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conversation_history')
    conv_count = cursor.fetchone()[0]
    st.metric("Total Conversations", conv_count)
    conn.close()

# Conversation History
st.subheader("💬 Recent Conversations")
for i, conv in enumerate(reversed(st.session_state["conversation_history"][-5:])):
    with st.expander(f"Conversation {len(st.session_state['conversation_history']) - i} - {conv['emotion'].title()} - {conv['timestamp'].strftime('%H:%M')}"):
        st.write(f"**You:** {conv['user']}")
        st.write(f"**{assistant_name}:** {conv['assistant']}")



# Footer
st.markdown("---")
st.markdown("🚀 **Features Active:** Conversational Memory | Agentic Tools | Natural Language Tasks | Simple Emotion Recognition | Background Listening | Email & Reminders")

# Instructions
with st.expander("📖 How to Use - ChatGPT-like Voice Assistant"):
    st.markdown("""
    ### 🎤 Voice Chat Modes:
    
    **🔄 True Continuous Voice Chat Mode (ChatGPT-like):**
    - Enable "Continuous Voice Chat Mode (ChatGPT-like)" in sidebar
    - Click "Start Continuous Chat" to begin
    - Speak naturally - the assistant responds and AUTOMATICALLY keeps listening
    - No need to click buttons between messages - truly continuous!
    - Say "stop listening", "goodbye", or click the stop button to exit
    - Perfect for natural flowing conversations just like ChatGPT voice mode
    
    **🎤 Single Voice Message Mode:**
    - Click "Voice Message" for one-time voice input
    - Get text or voice response based on your settings
    - Great for quick questions or commands
    
    **💬 Text Chat Mode:**
    - Enable "Text Input Mode" in sidebar
    - Type messages like in ChatGPT
    - Choose to get voice responses or text only
    
    ### 🗣️ Voice Commands Examples:
    - **Conversation**: "Hello, how are you?", "Tell me about artificial intelligence"
    - **Calculator**: "What's 15 times 23 plus 45?"
    - **Email**: "Send an email to john@example.com about tomorrow's meeting"
    - **Reminders**: "Remind me about the doctor appointment tomorrow at 2:30 PM"
    - **Weather**: "What's the weather like in London?" (requires API key)
    - **Search**: "Search for the latest news about AI technology"
    - **General Chat**: "Tell me a joke", "What's the meaning of life?", "Explain quantum physics"
    
    ### ⚙️ Setup:
    1. **Required**: Create a `.env` file with your API keys:
       - `GROQ_API_KEY=your_groq_key` (Required for AI responses)
       - `EMAIL_ADDRESS=your_email@gmail.com` (Optional - for email features)
       - `EMAIL_PASSWORD=your_app_password` (Optional - Gmail App Password)
       - `WEATHER_API_KEY=your_weather_key` (Optional - for weather queries)
    
    2. **Voice Settings**: 
       - Choose your preferred voice in the sidebar
       - Enable "Voice Responses" for spoken replies
    
    3. **Chat Modes**:
       - Try "Continuous Voice Chat" for natural conversations
       - Use "Voice Message" for single commands
    
    ### 💡 Tips:
    - Speak clearly and at normal pace
    - Wait for the "Listening..." indicator before speaking
    - In continuous mode, there's a brief pause between responses
    - You can switch between voice and text modes anytime
    - The assistant remembers your conversation context
    """)