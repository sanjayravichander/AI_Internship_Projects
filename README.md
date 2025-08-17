# 🚀 AI Internship Projects - Enterprise Dashboard

[![Railway](https://img.shields.io/badge/Railway-Deploy-purple)](https://railway.app/new/template?template=https://github.com/your-username/AI_Internship_Projects)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive collection of **9 production-ready AI applications** integrated into a single, enterprise-grade dashboard. Each application demonstrates different aspects of AI/ML engineering, from conversational AI to computer vision, predictive analytics, and intelligent document processing.

## 🌟 Live Demo

🔗 **[Deploy on Railway](https://railway.app/new/template?template=https://github.com/your-username/AI_Internship_Projects)**

## 📋 Applications Overview

### 🤖 1. Voice Assistant 2.0
- **Features**: Voice commands, email sending, reminders, calculator, weather, web search
- **Tech Stack**: LangChain, Groq, Speech Recognition, TTS, SQLite
- **Status**: ✅ Production Ready

### 📚 2. Document Intelligence Chatbot
- **Features**: Document upload, semantic search, entity extraction, knowledge graphs, analytics
- **Tech Stack**: LangChain, FAISS, spaCy, Plotly, NetworkX
- **Status**: ✅ Production Ready

### 🦠 3. COVID-19 Analytics Dashboard
- **Features**: Predictive modeling, anomaly detection, interactive charts, state comparisons
- **Tech Stack**: Plotly, Scikit-learn, Pandas, Groq AI
- **Status**: ✅ Production Ready

### 👋 4. Hand Gesture Recognition
- **Features**: Real-time detection, gesture classification, live camera feed
- **Tech Stack**: MediaPipe, OpenCV, Gradio
- **Status**: ✅ Production Ready

### 🎨 5. Cartoonify AI
- **Features**: Image cartoonification, video processing, multiple styles, AI analysis
- **Tech Stack**: OpenCV, ONNX, AnimeGAN, Groq Vision
- **Status**: ✅ Production Ready

### 😂 6. Meme Classification VLM
- **Features**: Zero-shot classification, AI explanations, vision-language processing
- **Tech Stack**: CLIP, Transformers, Groq LLM
- **Status**: ✅ Production Ready

### 📊 7. Student Report Card Generator
- **Features**: Data upload, grade calculation, visualizations, PDF reports
- **Tech Stack**: Pandas, Plotly, ReportLab, Streamlit
- **Status**: ✅ Production Ready

### 🧠 8. AI Quiz Game
- **Features**: AI-generated questions, multiple difficulty levels, score tracking, leaderboard
- **Tech Stack**: Groq LLM, Pandas, Streamlit
- **Status**: ✅ Production Ready

### 💭 9. Sentiment Analysis AI
- **Features**: Text sentiment analysis, emotion detection, confidence scoring
- **Tech Stack**: Transformers, VADER, TextBlob
- **Status**: ✅ Production Ready

## 🚀 Quick Start

### Option 1: Railway Deployment (Recommended)
1. Click the [Deploy on Railway](https://railway.app/new/template?template=https://github.com/your-username/AI_Internship_Projects) button
2. Connect your GitHub account and deploy with one click!

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/your-username/AI_Internship_Projects.git
cd AI_Internship_Projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm

# Run the application
streamlit run master_app_enterprise.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with your API keys:

```bash
# Required for AI features
GROQ_API_KEY=your_groq_api_key_here

# Optional but recommended
HUGGING_FACE_API_KEY=your_hf_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
WEATHER_API_KEY=your_weather_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### API Keys Setup
- **Groq API**: Get your free API key from [Groq Console](https://console.groq.com/)
- **Hugging Face**: Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
- **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/)

## 🏗️ Architecture

```
AI_Internship_Projects/
├── master_app_enterprise.py    # Main dashboard application
├── app.py                      # Hugging Face Spaces entry point
├── requirements.txt            # Python dependencies
├── Voice_Assistant_AI/         # Voice assistant application
├── Chatbot_AI/                # Document intelligence chatbot
├── COVID_19_AI/               # COVID-19 analytics dashboard
├── Hand_gesture_AI/           # Hand gesture recognition
├── Cartoonify_AI/             # Image cartoonification
├── Meme_Classification_VLM/   # Meme classification
├── Data_Handling/             # Student report generator
├── Python_Quiz_Game_AI/       # AI quiz game
└── Demo_files(Test-files)/    # Test files and demos
```

## 🎨 Features

### 🌟 Enterprise-Grade UI/UX
- **Modern Design**: Clean, professional interface with dark/light theme support
- **Responsive Layout**: Optimized for desktop, tablet, and mobile devices
- **Interactive Components**: Rich visualizations and real-time updates
- **Performance Optimized**: Fast loading times and efficient resource usage

### 🔒 Security & Privacy
- **Secure API Handling**: Environment variables for sensitive data
- **Input Validation**: Comprehensive validation for all user inputs
- **Error Handling**: Graceful error management with user-friendly messages
- **Data Privacy**: No data stored permanently, temporary processing only

### 📊 Analytics & Monitoring
- **Performance Metrics**: Real-time performance monitoring
- **Usage Analytics**: Application usage statistics
- **Error Tracking**: Comprehensive error logging and reporting
- **Memory Management**: Optimized for cloud deployment

## 🛠️ Technology Stack

### Core Frameworks
- **Streamlit**: Web application framework
- **LangChain**: AI application development
- **Transformers**: Machine learning models
- **PyTorch**: Deep learning framework

### AI/ML Libraries
- **Groq**: Fast LLM inference
- **OpenAI**: GPT models integration
- **Hugging Face**: Pre-trained models
- **spaCy**: Natural language processing
- **MediaPipe**: Computer vision
- **FAISS**: Vector similarity search

### Data & Visualization
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations
- **Matplotlib**: Static plotting
- **NetworkX**: Graph analysis

## 📱 Deployment Options

### 🚂 Railway
- **Generous Resources**: 8GB RAM, 8 vCPU on Hobby plan
- **Easy Setup**: One-click deployment from GitHub
- **Automatic Deployments**: Git-based continuous deployment
- **Custom Domains**: Free SSL certificates included

### 🌐 Other Platforms
- **Render**: Docker-based deployment
- **Streamlit Cloud**: Native Streamlit hosting
- **Heroku**: Container deployment
- **AWS/GCP/Azure**: Cloud platform deployment

## 🧪 Testing

```bash
# Run tests
pytest

# Run specific test file
pytest Demo_files(Test-files)/test_integration.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq**: For providing fast LLM inference
- **Railway**: For reliable and generous cloud hosting
- **Streamlit**: For the excellent web framework
- **OpenAI**: For GPT models and APIs
- **All Contributors**: Thank you for your contributions!

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/AI_Internship_Projects/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/AI_Internship_Projects/discussions)
- **Email**: your-email@example.com

---

<div align="center">
  <strong>🚀 Built with ❤️ for the AI community</strong>
  <br>
  <sub>Made by AI Intern | Powered by Streamlit & Railway</sub>
</div>