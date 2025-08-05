## 🧠 Meme Classifier using VLM + LLM
This project is a Streamlit web application that uses a two-stage AI pipeline to classify and explain memes. It leverages OpenAI's CLIP model for powerful zero-shot image classification and Groq's Llama 3 for witty, context-aware explanations.

## How It Works
The application follows a simple yet powerful workflow:

## Image Upload: The user uploads a meme image through the Streamlit interface.

VLM Classification (CLIP): The CLIP model processes the image and compares it against a predefined list of meme categories (e.g., "tech meme," "gaming meme," "AI meme"). It calculates a probability score for each category without being explicitly trained on them (zero-shot classification).

LLM Explanation (Groq): The top-predicted category and the probability scores are sent to the Groq API. A Llama 3 model then generates a humorous and insightful explanation for why the meme fits into that specific category.

Display Results: The application displays the original meme, the predicted category with its confidence score, and the detailed explanation from the LLM.

## ✨ Features
Accurate Zero-Shot Classification: Uses the CLIP-ViT-Large model to categorize memes without prior training.

Intelligent Explanations: Leverages the power of Groq's Llama 3 for fast and context-aware descriptions.

Simple Web Interface: Built with Streamlit for an easy-to-use and interactive experience.

Extensible Categories: The list of meme categories can be easily modified to suit different needs.

Fast Inference: Powered by the high-speed Groq API for near-instant LLM responses.

## 🛠️ Setup and Installation
Follow these steps to set up and run the project locally.
```bash
## 1. Clone the Repository
git clone https://github.com/your-username/meme-classifier.git
cd meme-classifier
```

## 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

```

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
Create a requirements.txt file with the following content:

streamlit
torch
transformers
python-dotenv
groq
requests
Pillow

## Then, install the required libraries using pip:
```bash
pip install -r requirements.txt
```
## 4. Set Up Environment Variables
You need a Groq API key for the LLM explanation part.

Get your free API key from the Groq Console.

Create a file named .env in the project's root directory.

Add your API key to the .env file like this:

GROQ_API_KEY="your-groq-api-key-here"

## Usage
Once you have completed the setup, you can run the Streamlit application with a single command:

```bash
streamlit run app.py
```

This will start the web server and open the application in your default web browser. You can now upload a meme and see the AI in action!

## Technologies Used
Framework: Streamlit

Vision Language Model (VLM): OpenAI CLIP

Large Language Model (LLM): Groq with Llama 3

Core Libraries: PyTorch, Transformers, Pillow