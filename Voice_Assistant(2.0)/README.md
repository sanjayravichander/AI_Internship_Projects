# 🎤 Voice Assistant 2.0 - Alice

## Overview

Alice is an advanced AI voice assistant integrated with LiveKit for real-time communication. This implementation provides a simple Streamlit UI that connects users directly to the LiveKit playground without interfering with the core agent functionality.

## Architecture

```
Voice_Assistant(2.0)/
├── agent.py                    # Core LiveKit agent implementation
├── voice_assistant_launcher.py # Streamlit UI launcher
├── config.py                   # Configuration settings
├── prompts_2.py               # AI prompts and instructions
├── tools_simple.py            # Assistant tools and functions
└── README.md                  # This file
```

## Features

### 🗣️ Core Capabilities
- **Natural Voice Conversation**: Fluid conversations using Google's LLM and TTS
- **System Control**: Open applications, take screenshots, execute commands
- **Web Integration**: Search web, get weather, access real-time information
- **Communication Hub**: Send emails, WhatsApp messages
- **Media Control**: Play YouTube videos, control media
- **Fast Startup**: Optimized LiveKit integration

### 🎯 UI Features
- **Simple Launch Interface**: Clean, professional Streamlit UI
- **Direct LiveKit Integration**: Seamless connection to playground
- **Configuration Management**: Easy customization via config.py
- **Status Monitoring**: Real-time system status display
- **Responsive Design**: Modern, mobile-friendly interface

## Integration with Master Dashboard

The Voice Assistant is integrated into the master enterprise dashboard (`master_app_enterprise.py`) as:

```python
"🤖 Voice Assistant 2.0": {
    "description": "Advanced AI assistant Alice with optimized LiveKit integration...",
    "function": None,
    "path": "Voice_Assistant(2.0)",
    "file": "voice_assistant_launcher.py",
    "tech_stack": "LiveKit Agents • Google LLM & TTS • System Control...",
    "features": ["Voice Conversation", "LiveKit Integration", "System Commands", ...],
    "status": "ready",
    "category": "AI Assistant",
    "complexity": "High",
    "last_updated": "2024-01-18"
}
```

## Configuration

### config.py Settings

```python
# LiveKit Playground URL
LIVEKIT_PLAYGROUND_URL = "https://agents-playground.livekit.io/"

# Assistant Configuration
ASSISTANT_NAME = "Alice"
ASSISTANT_VERSION = "2.0.0"
ASSISTANT_DESCRIPTION = "Advanced AI Assistant with LiveKit Integration"

# UI Theming
THEME_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
```

### Environment Variables

The agent requires these environment variables (in `.env` file):

```bash
GOOGLE_API_KEY=your_google_api_key
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
```

## Usage

### From Master Dashboard
1. Run the master dashboard: `streamlit run master_app_enterprise.py`
2. Select "🤖 Voice Assistant 2.0" from the sidebar
3. Click "🚀 Launch LiveKit Playground"
4. Allow microphone access when prompted
5. Start speaking with Alice

### Standalone
1. Navigate to Voice_Assistant(2.0) directory
2. Run: `streamlit run voice_assistant_launcher.py`
3. Follow the same steps as above

### Direct Agent (Console)
1. Navigate to Voice_Assistant(2.0) directory
2. Run: `python agent.py`
3. Agent will start in console mode with LiveKit

## Technical Details

### LiveKit Integration
- Uses LiveKit Agents framework for real-time communication
- Google's Gemini LLM for natural language processing
- Google's TTS for voice synthesis
- Noise cancellation with BVC (Background Voice Cancellation)

### Tools Available
- **Weather**: Get current weather information
- **Web Search**: Search the internet for information
- **Email**: Send emails through configured SMTP
- **System Commands**: Execute system-level commands
- **Application Control**: Open applications and websites
- **Screenshot**: Capture screen images
- **YouTube**: Play YouTube videos
- **WhatsApp**: Send WhatsApp messages

### Error Handling
- Network connectivity testing
- Retry logic for connection issues
- Graceful fallbacks for missing dependencies
- Comprehensive logging for debugging

## Customization

### Changing LiveKit Playground URL
Edit `config.py`:
```python
LIVEKIT_PLAYGROUND_URL = "https://your-custom-playground.com/"
```

### Modifying Assistant Name
Edit `config.py`:
```python
ASSISTANT_NAME = "YourAssistantName"
```

### UI Theming
Edit `config.py`:
```python
THEME_COLOR = "#your-primary-color"
SECONDARY_COLOR = "#your-secondary-color"
```

## Troubleshooting

### Common Issues
1. **Microphone Access**: Ensure browser allows microphone access
2. **API Keys**: Verify all required API keys are configured
3. **Network**: Check internet connectivity
4. **Dependencies**: Ensure all Python packages are installed

### Debug Mode
Enable debug logging in `agent.py`:
```python
logging.getLogger('tools_simple').setLevel(logging.DEBUG)
```

## Development

### Adding New Tools
1. Create function in `tools_simple.py`
2. Add to tools list in `agent.py`
3. Update prompts in `prompts_2.py` if needed

### Modifying UI
1. Edit `voice_assistant_launcher.py`
2. Update CSS styles as needed
3. Modify configuration in `config.py`

## License

Part of the AI Internship Projects collection. Built with ❤️ for educational and demonstration purposes.