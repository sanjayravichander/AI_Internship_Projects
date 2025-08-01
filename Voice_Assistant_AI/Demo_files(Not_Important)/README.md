# 🤖 ChatGPT-like Voice Assistant AI

An advanced AI voice assistant built with Python and Streamlit that provides ChatGPT-like conversational experience through voice and text interactions.

## 🌟 Key Features

### 🎤 Voice Chat Modes
- **Continuous Voice Chat**: ChatGPT-like experience with ongoing voice conversations
- **Single Voice Commands**: One-time voice input for quick queries
- **Text Chat**: Traditional text-based chat interface
- **Mixed Mode**: Combine voice input with text/voice responses

### 🧠 AI Capabilities
- **Conversational Memory**: Remembers context throughout the conversation
- **Agentic Tools**: Calculator, Email, Weather, Reminders, Web Search
- **Emotion Recognition**: Responds with appropriate emotional tone
- **Natural Language Processing**: Powered by Groq's Llama3-70B model

### 🔧 Advanced Features
- **Background Wake Word Detection**: Hands-free activation with "Hey Jarvis"
- **Multiple Voice Options**: Choose from available system voices
- **Persistent Memory**: Conversation history stored in SQLite database
- **Smart Reminders**: Schedule and manage reminders with natural language
- **Email Integration**: Send emails through voice commands
- **Weather Updates**: Get weather information for any city

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or download the project
cd Voice_Assistant_AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup
```bash
# Copy the example environment file
copy .env.example .env

# Edit .env file with your API keys:
# - GROQ_API_KEY (Required - get from https://console.groq.com/)
# - EMAIL_ADDRESS & EMAIL_PASSWORD (Optional - for email features)
# - WEATHER_API_KEY (Optional - get from https://openweathermap.org/api)
```

### 3. Run the Application
```bash
streamlit run voice_app_simple.py
```

## 💬 How to Use

### Continuous Voice Chat (ChatGPT-like)
1. Enable "Continuous Voice Chat Mode" in the sidebar
2. Click "Start Continuous Chat"
3. Speak naturally - the assistant will respond and keep listening
4. Say "stop listening" or "goodbye" to end the conversation

### Single Voice Commands
1. Click "Voice Message" button
2. Speak your question or command
3. Get instant response in text and/or voice

### Text Chat
1. Enable "Text Input Mode" in sidebar
2. Type your messages like in ChatGPT
3. Choose to receive voice responses or text only

## 🗣️ Example Conversations

**General Chat:**
- "Hello, how are you today?"
- "Tell me about artificial intelligence"
- "What's the meaning of life?"
- "Explain quantum physics in simple terms"

**Practical Commands:**
- "What's 15 times 23 plus 45?"
- "What's the weather in London?"
- "Send an email to john@example.com about tomorrow's meeting"
- "Remind me about the doctor appointment tomorrow at 2 PM"
- "Search for the latest news about AI"

## ⚙️ Configuration

### Voice Settings
- Choose from available system voices
- Enable/disable voice responses
- Adjust speech rate and volume

### Chat Modes
- **Continuous Mode**: For natural conversations
- **Single Command Mode**: For quick queries
- **Text Mode**: For typing instead of speaking
- **Mixed Mode**: Combine different input/output methods

### Advanced Options
- Background wake word detection
- Conversation memory settings
- Emotion recognition
- Tool integrations

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **AI Model**: Groq Llama3-70B via LangChain
- **Speech Recognition**: Google Speech Recognition
- **Text-to-Speech**: pyttsx3 with system voices
- **Database**: SQLite for conversation history and reminders
- **Scheduling**: APScheduler for reminders

### Tools & Integrations
- **Calculator**: Mathematical computations
- **Email**: Gmail SMTP integration
- **Weather**: OpenWeatherMap API
- **Web Search**: Simulated search (extensible)
- **Reminders**: Natural language scheduling

## 🔒 Privacy & Security

- All conversations stored locally in SQLite database
- API keys stored securely in .env file
- No data sent to third parties except for AI processing
- Voice processing happens locally when possible

## 🤝 Contributing

Feel free to contribute by:
- Adding new tools and integrations
- Improving voice recognition accuracy
- Enhancing the user interface
- Adding new features and capabilities

## 📝 License

This project is open source and available under the MIT License.

---

**Enjoy your ChatGPT-like voice assistant experience! 🎉**