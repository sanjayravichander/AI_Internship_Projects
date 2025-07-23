---
description: Repository Information Overview
alwaysApply: true
---

# Python Quiz Game AI Information

## Summary
This project is an AI-powered quiz game built with Python and Streamlit. It uses the Groq API through LangChain to generate dynamic quiz questions on any topic. The application allows users to take quizzes in two modes: LLM-generated questions or from uploaded CSV files. It features difficulty levels, score tracking, explanations, and a leaderboard system.

## Structure
- **Python_Quiz_Game_AI/**: Main project directory containing application code
  - `ai_quiz.py`: Primary implementation of the AI quiz game
  - `quiz.py`: Alternative implementation with similar functionality
  - `research.ipynb`: Empty notebook for development/testing
- **Root Directory**:
  - `app.py`: Empty file (possibly for future development)
  - `requirements.txt`: Dependencies list

## Language & Runtime
**Language**: Python
**Version**: 3.10.18 (identified from notebook metadata)
**Framework**: Streamlit (web application framework)
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- langchain: Framework for LLM application development
- langchain_community: Community extensions for LangChain
- langchain_groq: Groq integration for LangChain
- groq: API client for Groq LLM service
- streamlit: Web application framework
- pandas: Data manipulation library
- numpy: Numerical computing library
- python-dotenv (implied): Environment variable management

## Build & Installation
```bash
pip install -r requirements.txt
```

## Usage
**Run the Application**:
```bash
streamlit run Python_Quiz_Game_AI/ai_quiz.py
```

**Environment Setup**:
The application requires a Groq API key stored in a `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

## Features
**Quiz Modes**:
- LLM Generated Quiz: Creates questions on any topic using Groq's Gemma2-9b-it model
- CSV Upload: Uses questions from a user-provided CSV file

**Difficulty Levels**:
- Beginner
- Intermediate
- Advanced

**Question Types**:
- Multiple-choice (A, B, C, D options)
- Boolean (True/False)

**User Experience**:
- Personalized with user name
- Score tracking and percentage calculation
- Explanations for correct answers
- Downloadable results in CSV format
- Persistent leaderboard

## Data Handling
**CSV Format Support**:
- Flexible column naming with aliases
- Automatic detection of boolean questions
- Support for question, answer, options, explanation, and difficulty columns

## Integration
**LLM Integration**:
- Uses Groq's Gemma2-9b-it model
- Custom prompts for question generation and explanations
- Temperature setting of 0.9 for creative variety