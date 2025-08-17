import os
# Force PyTorch backend and disable TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from groq import Groq

# ---------------------- Load .env from project root ----------------------
# Try multiple paths to find the .env file
env_paths = [
    os.path.join(os.path.dirname(__file__), '..', '.env'),  # Parent directory
    os.path.join(os.path.dirname(os.getcwd()), '.env'),     # Original path
    '.env',  # Current directory
    os.path.join(os.getcwd(), '.env')  # Current working directory
]

for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        break
else:
    # If no .env file found, try loading from environment
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------- Groq Client Setup ----------------------
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found! Please check your .env file.")
    st.info("💡 Make sure the .env file contains: GROQ_API_KEY=your_api_key_here")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------- Load CLIP model with Streamlit caching ----------------------
@st.cache_resource
def load_clip_model():
    """Load CLIP model with Streamlit caching to avoid re-downloading"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-large-patch14"
    
    # Use a cache directory for models
    model_cache_dir = "models/clip"
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Download and load models (cached by Streamlit)
    model = CLIPModel.from_pretrained(model_id, cache_dir=model_cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=model_cache_dir)
    
    return model, processor, device

# Load models using cached function
model, processor, device = load_clip_model()

# ---------------------- Define meme categories ----------------------
categories = [
    "tech meme", "coding meme", "startup meme", "finance meme",
    "college meme", "politics meme", "sports meme", "relationship meme",
    "dark humor meme", "wholesome meme", "sarcastic meme", "relatable meme",
    "pop culture meme", "motivational meme", "gaming meme", "AI meme"
]

# ---------------------- Predict Category Using CLIP ----------------------
def predict_category(image: Image.Image, candidate_labels: list):
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    top_idx = probs.argmax()
    top_label = candidate_labels[top_idx]
    top_prob = round(probs[top_idx] * 100, 2)

    return top_label, top_prob, dict(zip(candidate_labels, probs))

# ---------------------- Get Explanation from Groq LLM ----------------------
def get_explanation(label: str, label_prob: float, other_probs: dict):
    formatted_probs = "\n".join(
        [f"- {cat}: {round(prob * 100, 2):.2f}%" for cat, prob in sorted(other_probs.items(), key=lambda x: x[1], reverse=True)]
    )

    prompt = f"""
You're a meme classification expert. A meme image has been predicted as:

✅ Predicted Category: **{label}** ({label_prob:.2f}% confidence)

Here are the top category probabilities:
{formatted_probs}

📷 Based on the image's visual content, explain why it is categorized as a **{label}** meme.
Keep the tone witty and insightful like a meme analyst.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a meme explanation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"⚠️ Error getting explanation: {e}"

    return explanation

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Meme Classifier AI", layout="centered")
st.title("🧠 Meme Classifier using VLM + LLM")
st.markdown("Upload a meme and I'll guess the category using CLIP + explain it using Groq LLM 🔍")

uploaded_file = st.file_uploader("📤 Upload Meme Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Meme", use_container_width=True)

    with st.spinner("Analyzing meme..."):
        label, prob, prob_dict = predict_category(image, categories)
        explanation = get_explanation(label, prob, prob_dict)

    st.markdown(f"### ✅ Predicted Category: `{label}` ({prob:.2f}%)")
    st.markdown("### 💬 LLM Explanation")
    st.markdown(explanation)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("Built with ❤️ using [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) + [Groq](https://groq.com/) + Streamlit")
