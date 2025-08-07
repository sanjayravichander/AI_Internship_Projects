Here’s a polished and well-structured README.md for your Voice Assistant AI, based on your outline and best practices for open-source projects:

---

# 🤖 Advanced AI Voice Assistant with Agentic Capabilities

A powerful, real-time voice assistant built with **Streamlit**, **LangChain**, **Groq**, and **SpeechRecognition**. It supports natural conversations, voice interaction, tool usage, reminders, and agentic reasoning with persistent memory.

---

## 🚀 Features

- 🎙️ **Voice Input & Output**  
  Both single-turn and continuous (ChatGPT-style) conversations with emotion-aware speech synthesis.

- 🧠 **Conversational Memory**  
  Remembers the last 10 interactions for context using LangChain memory.

- 📚 **Integrated Tools**  
  - ✅ Calculator  
  - 🔍 Web Search (simulated)  
  - 📧 Email Sending  
  - ⏰ Reminders  
  - 🌦️ Weather Lookup  

- 🔔 **Wake Word Detection**  
  Passive listening for activation words (e.g., “Hey Jarvis”).

- 🎛️ **User Preferences & Persistent Memory**  
  Stores settings, reminders, chat history, and user details in `assistant_memory.db`.

- 📡 **Web-Based Interface**  
  Streamlit dashboard for voice settings, chat history, and real-time stats.

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sanjayravichander/AI_Internship_Projects.git
cd AI_Internship_Projects/Voice_Assistant_AI
```

### 2. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with the following keys:

```ini
GROQ_API_KEY=your_groq_api_key_here
EMAIL_ADDRESS=your_email_address@gmail.com
EMAIL_PASSWORD=your_gmail_app_password  # Use an app password, not your Gmail password
WEATHER_API_KEY=your_openweather_api_key
```

---

## 🧪 Running the Application

```bash
python voice_app_simple.py
```

- Access the app at: [http://localhost:8501](http://localhost:8501)

---

## 🎤 Usage Tips

- Use “Voice Message” for single-turn interactions.
- Enable “Continuous Chat Mode” for multi-turn conversations.
- Enable “Voice Responses” for spoken replies.
- Example commands:
  - “What’s the weather in London?”
  - “Remind me about the meeting tomorrow at 3 PM”
  - “Send an email to alex@example.com | Subject | Hello there!”
  - “Open YouTube” or “Play lo-fi beats on YouTube”

---

## 📦 Data Storage

Data is stored in an SQLite database: `assistant_memory.db`

- **Tables:**
  - `user_preferences`
  - `conversation_history`
  - `reminders`

---

## 🧠 Built With

- LangChain
- Groq LLM API
- Streamlit
- SpeechRecognition
- pyttsx3
- APScheduler
- SQLite3

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Sanjay**  
Data Science & AI Enthusiast

**Contact:**  
For collaboration or queries, reach out via [LinkedIn](#) or email!

---

## ✅ requirements.txt

```
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
sqlite3
```

> **Note:**  
> If you encounter issues installing `pyaudio`, on Windows run:
>
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```
