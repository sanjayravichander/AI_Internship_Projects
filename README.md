That's an excellent Streamlit quiz application\! It's well-structured, handles both LLM-generated and CSV-uploaded questions, and includes great features like difficulty mapping, explanation generation, and result tracking.

Here's an impressive `README.md` for your project, designed to be clear, engaging, and highlight its key features:

-----

# 🧠 AI Quiz Game

Test your knowledge with real-time, AI-generated questions or challenge yourself with quizzes from your own CSV files\! This interactive quiz application, built with Streamlit, leverages the power of Large Language Models (LLMs) to create dynamic and engaging learning experiences.

-----

## ✨ Features

  * **Dual Quiz Modes:** Choose between **LLM Generated Quizzes** on any topic or **Upload CSV File** to use your own custom question sets.
  * **Dynamic Question Generation:** Utilizes the **Groq API** and a `gemma2-9b-it` LLM to create unique, multiple-choice questions in real-time.
  * **Difficulty Levels:** Select `Beginner`, `Intermediate`, or `Advanced` difficulty for LLM-generated quizzes. CSV uploads can also leverage difficulty levels if present.
  * **Smart CSV Handling:**
      * **Flexible Column Detection:** Automatically maps common column names (e.g., `question`, `answer`, `a`, `optiona`, `explanation`, `difficulty`).
      * **Boolean Question Support:** Intelligently identifies and formats True/False questions from CSVs.
      * **Difficulty Mapping:** Converts various difficulty aliases (e.g., "easy", "hard", "1", "L3") into standardized levels.
      * **Explanation Generation:** Automatically generates explanations for CSV questions if none are provided.
      * **Question Uniqueness:** Filters out previously used questions from CSVs to ensure fresh quizzes.
  * **Interactive User Interface:** Clear and intuitive Streamlit UI for seamless quiz progression.
  * **Score Tracking & Results Summary:** View your score, percentage, and receive performance feedback at the end of each quiz.
  * **Detailed Question Review:** Expandable sections for each question in the results, showing your answer, the correct answer, and a comprehensive explanation.
  * **Downloadable Results:** Export your quiz results to a CSV file for review or record-keeping.
  * **Session Management:** Intelligent handling of Streamlit session state to maintain quiz progress and avoid question repetition across restarts.

-----

## 🚀 How to Run Locally

Follow these steps to get your AI Quiz Game up and running on your local machine.

### Prerequisites

  * **Python 3.8+**
  * **Groq API Key:** You'll need an API key from [Groq Cloud](https://console.groq.com/keys).

### Setup Instructions

1.  **Clone the Repository (or create the files):**
    If you have a Git repository:

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

    Otherwise, save the provided code as `quiz_app.py` and create the following files/folders.

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    ```

    Activate the virtual environment:

      * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
      * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in your project root with the following content:

    ```
    streamlit
    pandas
    python-dotenv
    langchain-groq
    langchain-core
    ```

    Then, install them:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your Groq API Key:**
    Create a file named `.env` in the root of your project directory (the same directory as `quiz_app.py`).
    Add your Groq API key to this file:

    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    ```

    Replace `"YOUR_GROQ_API_KEY_HERE"` with your actual key from Groq Cloud.

5.  **Run the Streamlit App:**

    ```bash
    streamlit run quiz_app.py
    ```

    Your browser should automatically open to the Streamlit application (usually `http://localhost:8501`).

-----

## 📁 CSV File Format for Uploads

If you choose the "Upload CSV File" mode, your CSV should ideally contain the following columns (column names are flexible due to the intelligent mapping):

| Column Aliases (examples) | Description                                                                            | Example Value                                |
| :------------------------ | :------------------------------------------------------------------------------------- | :------------------------------------------- |
| `question`, `text`        | The main question text.                                                                | What is the capital of France?               |
| `a`, `optiona`            | Option A                                                                               | Paris                                        |
| `b`, `optionb`            | Option B                                                                               | Berlin                                       |
| `c`, `optionc`            | Option C                                                                               | Rome                                         |
| `d`, `optiond`            | Option D                                                                               | Madrid                                       |
| `answer`, `correct`       | The correct option letter (A, B, C, D) or 'T'/'F' for boolean questions.             | A                                            |
| `explanation`, `reason`   | An optional explanation for the correct answer. Will be generated by LLM if missing. | Paris is the capital and most populous city... |
| `difficulty`, `level`     | Optional: Difficulty level (e.g., "Beginner", "Intermediate", "Advanced", "Easy").   | Beginner                                     |

**Example `my_quiz.csv`:**

```csv
question,a,b,c,d,answer,explanation,difficulty
What is the largest ocean on Earth?,Atlantic,Indian,Pacific,Arctic,C,The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It spans one-third of the globe.,Intermediate
Is the Earth flat?,True,False,,B,The Earth is an oblate spheroid.,Beginner
Which programming language is known as the "language of the web"?,Python,Java,JavaScript,C++,C,JavaScript is a high-level, often just-in-time compiled language that conforms to the ECMAScript standard. It is a multi-paradigm, prototype-based, object-oriented, and dynamic language, primarily used to enable interactive web pages.,Beginner
What is the process of converting AC to DC called?,Rectification,Amplification,Modulation,Demodulation,A,Rectification is the process of converting an alternating current (AC) to a direct current (DC). This is typically achieved using a rectifier, which is an electrical device consisting of one or more PN junction diodes.,Advanced
```

-----

## 🤝 Contributing

Feel free to fork this repository, open issues, and submit pull requests. Contributions are welcome to enhance features, improve the UI, or expand the LLM capabilities.

-----

## 📄 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

-----

## 🙏 Acknowledgements

  * Built with [Streamlit](https://streamlit.io/)
  * Powered by [Groq](https://groq.com/) for fast LLM inference
  * [LangChain](https://www.langchain.com/) for LLM orchestration

-----
