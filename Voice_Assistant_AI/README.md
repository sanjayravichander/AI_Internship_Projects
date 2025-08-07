# 🤖 Advanced AI Voice Assistant with Agentic Capabilities

This project is a powerful, real-time voice assistant built using **Streamlit**, **LangChain**, **Groq**, and **Speech Recognition**. It supports **natural conversation**, **voice interaction**, **tool usage**, and **reminders**, and is capable of performing reasoning-based tasks with memory.

---

## 🚀 Features

- 🎙️ **Voice Input + Voice Output**  
  Supports both continuous (ChatGPT-style) and single-turn voice conversations with emotion-aware speech synthesis.

- 🧠 **Conversational Memory**  
  Remembers the last 10 interactions for better context using LangChain’s memory system.

- 📚 **Tool Integration**  
  Equipped with tools like:
  - ✅ Calculator
  - 🔍 Web Search (simulated)
  - 📧 Email Sending
  - ⏰ Reminder Scheduling
  - 🌦️ Weather Lookup

- 🔔 **Background Wake Word Detection**  
  Passive listening for wake words (e.g., "Hey Jarvis") to activate the assistant.

- 🎛️ **User Preferences & Memory Storage**  
  Stores voice settings, reminders, chat history, and user details in `assistant_memory.db`.

- 📡 **Web-Based Interface**  
  Streamlit dashboard with voice settings, chat history, and real-time stats.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-voice-assistant.git
cd ai-voice-assistant
2. Set up a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
📁 Environment Variables
Create a .env file in the root directory with the following keys:

ini
Copy
Edit
GROQ_API_KEY=your_groq_api_key_here
EMAIL_ADDRESS=your_email_address@gmail.com
EMAIL_PASSWORD=your_gmail_app_password  # not your Gmail password
WEATHER_API_KEY=your_openweather_api_key
🧪 Running the Application
bash
Copy
Edit
python voice_app_simple.py
Access the app via http://localhost:8501 in your browser.

🎤 Usage Tips
Use “Voice Message” for single-turn interactions.

Enable “Continuous Chat Mode” for ChatGPT-style conversation.

Enable "Voice Responses" for spoken replies.

Say things like:

"What's the weather in London?"

"Remind me about the meeting tomorrow at 3 PM"

"Send an email to alex@example.com | Subject | Hello there!"

"Open YouTube" or "Play lo-fi beats on YouTube"

📦 Data Storage
Data is stored in an SQLite database: assistant_memory.db
Tables:

user_preferences

conversation_history

reminders

🧠 Built With
LangChain

Groq LLM API

Streamlit

SpeechRecognition

pyttsx3

APScheduler

SQLite3

📄 License
This project is licensed under the MIT License.

🙋‍♂️ Author
Sanjay
Data Science & AI Enthusiast

📬 Contact
For collaboration or queries, feel free to reach out via LinkedIn or email!

yaml
Copy
Edit

---

### ✅ `requirements.txt`

```txt
streamlit
speechrecognition
pyttsx3
pyaudio
apscheduler
langchain
langchain-groq
langchain-core
python-dotenv
openai
requests
email-validator
python-dateutil
pytz
sqlite3  # (comes built-in with Python but included for clarity)
✅ Note: Ensure pyaudio is installed properly. On Windows, install it via:

bash
Copy
Edit
pip install pipwin
pipwin install pyaudio
