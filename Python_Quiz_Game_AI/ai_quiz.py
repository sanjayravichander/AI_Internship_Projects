import streamlit as st
import pandas as pd
import os
import json
import re
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Helper function to clean answers
def clean_answer(answer):
    """Extract just the letter from answer string"""
    if isinstance(answer, str):
        # Extract the first A-D letter or T/F
        match = re.search(r'[A-DTtFf]', answer.upper())
        if match:
            letter = match.group(0)
            # Convert T/F to A/B for boolean questions
            if letter == 'T': return 'A'
            if letter == 'F': return 'B'
            return letter
    return "X"

# Session State Initialization
for key, default in {
    'score': 0,
    'qn': 0,
    'results': [],
    'total_questions': 5,
    'answered': False,
    'question_cache': set(),
    'used_questions': [],
    'quiz_started': False,
    'user_name': "",
    'uploaded_df': None,
    'show_explanations': False,
    'difficulty_mapping': {
        # Beginner level aliases
        'easy': 'Beginner', 'easiest': 'Beginner', 'beginner': 'Beginner', 'novice': 'Beginner',
        'basic': 'Beginner', 'simple': 'Beginner', 'elementary': 'Beginner', 'entry': 'Beginner',
        'entry-level': 'Beginner', 'introductory': 'Beginner', 'starter': 'Beginner', '1': 'Beginner',
        'level 1': 'Beginner', 'l1': 'Beginner', 'e': 'Beginner', 'a1': 'Beginner', 'a': 'Beginner',
        
        # Intermediate level aliases
        'medium': 'Intermediate', 'intermediate': 'Intermediate', 'moderate': 'Intermediate',
        'average': 'Intermediate', 'standard': 'Intermediate', 'regular': 'Intermediate',
        'middle': 'Intermediate', 'mid': 'Intermediate', 'normal': 'Intermediate', '2': 'Intermediate',
        'level 2': 'Intermediate', 'l2': 'Intermediate', 'm': 'Intermediate', 'b': 'Intermediate',
        'b1': 'Intermediate', 'b2': 'Intermediate',
        
        # Advanced level aliases
        'hard': 'Advanced', 'advanced': 'Advanced', 'expert': 'Advanced', 'difficult': 'Advanced',
        'challenging': 'Advanced', 'complex': 'Advanced', 'tough': 'Advanced', 'complicated': 'Advanced',
        'high': 'Advanced', 'upper': 'Advanced', 'professional': 'Advanced', '3': 'Advanced',
        'level 3': 'Advanced', 'l3': 'Advanced', 'h': 'Advanced', 'c': 'Advanced', 'c1': 'Advanced',
        'c2': 'Advanced', 'd': 'Advanced'
    }
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# LLM Initialization
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.9,
    api_key=groq_api_key,
)

prompt_template = ChatPromptTemplate.from_template("""
Generate a {difficulty} level multiple-choice quiz question on the topic of "{topic}".

Difficulty Guidelines:
- Beginner: Simple recall questions, basic concepts, straightforward facts
- Intermediate: Application of concepts, moderate complexity, some analysis required
- Advanced: Complex scenarios, deep understanding required, challenging concepts

Ensure the question is unique and not similar to any of these previously used questions:
{used_questions}

Requirements:
1. Focus on a different sub-topic or concept than previous questions
2. Include novel scenarios or applications
3. The question text should be distinct from any previously generated questions
4. Include four plausible options (A, B, C, D) with only one correct answer
5. Make distractors (wrong options) plausible but clearly incorrect upon careful consideration
6. Provide a thorough explanation that teaches the concept, not just states the answer
7. Ensure the difficulty level matches the requested {difficulty} level

Format:
Question: [Unique question text here]
Options:
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]
Answer: [Correct letter A/B/C/D]
Explanation: [Detailed explanation that teaches the concept]
""")

explanation_prompt = ChatPromptTemplate.from_template("""
Given this quiz question and correct answer, provide a detailed educational explanation:

Question: {question}
Correct Answer: {correct_answer}

Your explanation should:
1. Explain why the correct answer is right
2. Clarify any misconceptions related to wrong answers
3. Provide additional context or background information
4. Include relevant facts, principles, or theories
5. Be educational and help the user learn from this question

Explanation:
""")

# Streamlit UI
st.title("AI Quiz Game")
st.markdown("Test your knowledge with real-time questions!")

if not st.session_state.quiz_started:
    st.session_state.user_name = st.text_input("What's your name?", value=st.session_state.user_name)
    if not st.session_state.user_name.strip():
        st.stop()
    else:
        st.markdown(f"Hi **{st.session_state.user_name}**, please set your quiz preferences.")

quiz_mode = st.radio("Select Quiz Mode:", 
                    ["LLM Generated Quiz", "Upload CSV File"],
                    index=0)

if quiz_mode == "Upload CSV File":
    st.session_state.uploaded_file = st.file_uploader("Upload Quiz CSV File", type=["csv"])

# Quiz setup form
with st.form("quiz_setup"):
    if quiz_mode == "LLM Generated Quiz":
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
        with col2:
            topic = st.text_input("Enter Quiz Topic", value="General Knowledge")
        num_questions = st.slider("Number of Questions", 1, 25, value=5)
        st.session_state.quiz_mode = "llm"
    else:
        topic = "Uploaded Questions"
        difficulty = "From CSV"
        num_questions = st.slider("Number of Questions to Use", 1, 50, value=10)
        st.session_state.quiz_mode = "csv"

    start = st.form_submit_button("Start Quiz")

# Start quiz processing
if start:
    if st.session_state.quiz_mode == "csv" and not st.session_state.get("uploaded_file"):
        st.warning("Please upload a CSV file to continue")
        st.stop()
    
    if st.session_state.quiz_mode == "llm" and not topic.strip():
        st.warning("Please enter a quiz topic")
        st.stop()

    st.session_state.difficulty = difficulty
    st.session_state.topic = topic
    st.session_state.total_questions = num_questions

    if st.session_state.quiz_mode == "csv":
        try:
            df = pd.read_csv(st.session_state.uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            
            # Enhanced column detection with aliases
            column_aliases = {
                'question': ['question', 'text', 'prompt', 'query', 'problem'],
                'answer': ['answer', 'correct', 'solution', 'key', 'right'],
                'a': ['a', 'optiona', 'choicea', '1', 'one', 'true'],
                'b': ['b', 'optionb', 'choiceb', '2', 'two', 'false'],
                'c': ['c', 'optionc', 'choicec', '3', 'three'],
                'd': ['d', 'optiond', 'choiced', '4', 'four'],
                'explanation': ['explanation', 'reason', 'rationale', 'justification'],
                'difficulty': ['difficulty', 'level', 'complexity', 'grade']
            }
            
            # Find matching columns
            column_map = {}
            for standard_name, aliases in column_aliases.items():
                for alias in aliases:
                    if alias in df.columns:
                        column_map[standard_name] = alias
                        break
            
            # Apply column mapping
            df = df.rename(columns={v: k for k, v in column_map.items()})
            
            # Handle difficulty mapping
            if 'difficulty' in df.columns:
                # Convert to lowercase and strip whitespace
                df['difficulty'] = df['difficulty'].astype(str).str.lower().str.strip()
                
                # Map difficulty values using our extensive mapping
                df['difficulty'] = df['difficulty'].map(
                    lambda x: st.session_state.difficulty_mapping.get(x, "Intermediate")
                )
                
                # Show unique difficulty levels found in the CSV
                unique_difficulties = df['difficulty'].unique()
                st.info(f"Difficulty levels found in CSV: {', '.join(unique_difficulties)}")
            else:
                # If no difficulty column, use the selected difficulty from the form
                df['difficulty'] = difficulty
                st.info(f"No difficulty column found in CSV. Using selected difficulty: {difficulty}")
            
            # Ensure required columns exist
            if 'question' not in df.columns:
                df['question'] = df.iloc[:, 0]  # Use first column as question
            
            if 'answer' not in df.columns:
                # Try to find answer in options columns
                for col in ['a', 'b', 'c', 'd']:
                    if col in df.columns:
                        df['answer'] = col.upper()
                        break
                else:
                    df['answer'] = "A"  # Default to A if no answer found
            
            # Clean answers
            df["answer"] = df["answer"].apply(clean_answer)
            
            # Improved Boolean Question Detection
            def detect_boolean_question(row):
                # Check if options contain true/false values
                options = []
                for col in ['a', 'b']:
                    if col in row and not pd.isna(row[col]):
                        options.append(str(row[col]).strip().lower())
                
                # Check if question text suggests true/false
                question_text = str(row.get('question', '')).lower()
                
                # Check for explicit true/false options
                has_true_false_options = (
                    len(options) >= 2 and
                    'true' in options[0] and 'false' in options[1]
                ) or (
                    len(options) >= 2 and
                    'false' in options[0] and 'true' in options[1]
                )
                
                # Check for true/false in question text
                has_true_false_question = (
                    'true or false' in question_text or
                    'true/false' in question_text or
                    question_text.startswith('is it true that') or
                    question_text.startswith('true or false:')
                )
                
                # Check answer format
                answer_is_tf = (
                    'answer' in row and 
                    str(row['answer']).strip().upper() in ['T', 'F', 'TRUE', 'FALSE', 'A', 'B']
                )
                
                return has_true_false_options or has_true_false_question or answer_is_tf

            # First detect boolean questions
            df['is_boolean'] = df.apply(detect_boolean_question, axis=1)
            
            # Then fill missing options only for non-boolean questions
            for idx, row in df.iterrows():
                if not row['is_boolean']:
                    for letter in ['a', 'b', 'c', 'd']:
                        if letter not in df.columns or pd.isna(row.get(letter)):
                            df.at[idx, letter] = f"Option {letter.upper()}"
                else:
                    # For boolean questions, ensure we have True/False options
                    if 'a' not in df.columns or pd.isna(row.get('a')):
                        df.at[idx, 'a'] = "True"
                    if 'b' not in df.columns or pd.isna(row.get('b')):
                        df.at[idx, 'b'] = "False"
            
            # Filter by difficulty if specified in the form
            if difficulty != "From CSV" and 'difficulty' in df.columns:
                # Get questions matching the selected difficulty
                matching_difficulty = df[df['difficulty'] == difficulty]
                
                # If we have enough questions of the selected difficulty, use only those
                if len(matching_difficulty) >= num_questions:
                    df = matching_difficulty
                    st.success(f"Using {len(df)} questions with {difficulty} difficulty")
                else:
                    st.warning(f"Only {len(matching_difficulty)} questions with {difficulty} difficulty. Using all difficulty levels.")
            
            # Filter out previously used questions
            all_questions = df['question'].tolist()
            new_questions = [q for q in all_questions if q not in st.session_state.used_questions]
            
            # Shuffle the questions thoroughly
            random.shuffle(new_questions)
            
            if len(new_questions) < num_questions:
                st.warning(f"Only {len(new_questions)} unique questions available. Using all.")
                # If we don't have enough new questions, use all available new ones
                filtered_df = df[df['question'].isin(new_questions)]
            else:
                # Select the requested number of questions
                selected_questions = new_questions[:num_questions]
                filtered_df = df[df['question'].isin(selected_questions)]
            
            # Shuffle again for good measure
            st.session_state.uploaded_df = filtered_df.sample(frac=1).reset_index(drop=True)
            st.session_state.total_questions = len(st.session_state.uploaded_df)
            
            # Add to used questions
            st.session_state.used_questions.extend(st.session_state.uploaded_df['question'].tolist())
            
            # Show how many questions are left for future quizzes
            remaining = len(all_questions) - len(st.session_state.used_questions)
            if remaining > 0:
                st.info(f"{remaining} unique questions remaining in the CSV for future quizzes")
        
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            st.stop()
    else:
        st.session_state.uploaded_df = None

    st.session_state.quiz_started = True
    st.rerun()

if not st.session_state.quiz_started:
    st.stop()

def generate_explanation(question, correct_answer):
    """Generate explanation using LLM if not provided"""
    chain = explanation_prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "correct_answer": correct_answer
    })

# Improved parsing with regex
def parse_question(raw_text):
    try:
        # Extract question
        question_match = re.search(r'Question:\s*(.+?)(?:\nOptions:|$)', raw_text, re.DOTALL)
        q = question_match.group(1).strip() if question_match else "Could not parse question"
        
        # Extract options
        options = []
        options_section = re.search(r'Options:\s*(.+?)(?:\nAnswer:|$)', raw_text, re.DOTALL)
        if options_section:
            options_text = options_section.group(1)
            for letter in ['A', 'B', 'C', 'D']:
                option_match = re.search(fr'{letter}\.\s*(.+?)(?:\n[A-D]\.|$)', options_text, re.DOTALL)
                if option_match:
                    options.append(f"{letter}. {option_match.group(1).strip()}")
                else:
                    options.append(f"{letter}. Option not parsed")
        else:
            options = [f"{letter}. Option not parsed" for letter in ['A', 'B', 'C', 'D']]
        
        # Extract answer
        answer_match = re.search(r'Answer:\s*([A-D])', raw_text)
        ans = clean_answer(answer_match.group(1)) if answer_match else "X"
        
        # Extract explanation
        explanation_match = re.search(r'Explanation:\s*(.+)', raw_text, re.DOTALL)
        exp = explanation_match.group(1).strip() if explanation_match else None
        
        return q, options, ans, exp
    except Exception as e:
        return f"Error: {str(e)}", [], "X", None

# Main Quiz Logic
if st.session_state.qn < st.session_state.total_questions:
    with st.spinner("Generating question..."):
        key = f"qdata_{st.session_state.qn}"
        if key not in st.session_state:
            if st.session_state.uploaded_df is not None:
                # Use CSV data
                row = st.session_state.uploaded_df.iloc[st.session_state.qn]
                question = row["question"]
                answer = clean_answer(row["answer"])
                
                # Get options - handle boolean questions differently
                options = []
                is_boolean = False
                
                # Check if it's a boolean question
                if 'is_boolean' in row and row['is_boolean']:
                    is_boolean = True
                    # For boolean, only show True/False options
                    options.append(f"A. True")
                    options.append(f"B. False")
                    # Map T/F answers to A/B
                    if answer == 'T': answer = 'A'
                    if answer == 'F': answer = 'B'
                else:
                    # For multiple-choice, show all options
                    for letter in ['a', 'b', 'c', 'd']:
                        if letter in row and not pd.isna(row[letter]):
                            options.append(f"{letter.upper()}. {row[letter]}")
                        else:
                            # Only add placeholder if we're not in a boolean question
                            options.append(f"{letter.upper()}. Option {letter.upper()}")
                
                # Get or generate explanation
                if "explanation" in row and pd.notna(row["explanation"]) and str(row["explanation"]).strip():
                    explanation = row["explanation"]
                else:
                    # Generate a more detailed explanation based on the question type
                    if is_boolean:
                        # For boolean questions, include the correct answer in the prompt
                        correct_answer = "True" if answer == "A" else "False"
                        explanation = generate_explanation(
                            f"True or False: {question}", 
                            f"The answer is {correct_answer}"
                        )
                    else:
                        # For multiple choice, include the options in the prompt
                        option_text = "\n".join(options)
                        correct_option = next((opt for opt in options if opt.startswith(f"{answer}. ")), "")
                        explanation = generate_explanation(
                            f"{question}\n{option_text}", 
                            f"The correct answer is {correct_option}"
                        )
                
                # Get difficulty if available
                difficulty = row["difficulty"] if "difficulty" in row else "Not Specified"
                
                st.session_state[key] = {
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "explanation": explanation,
                    "difficulty": difficulty,
                    "is_boolean": is_boolean
                }
            else:
                # Generate new question with LLM
                max_attempts = 8  # Increase retry attempts for better uniqueness
                for attempt in range(max_attempts):
                    with st.spinner(f"Generating question (attempt {attempt+1}/{max_attempts})..."):
                        # Get the most recent used questions to avoid repetition
                        recent_questions = st.session_state.used_questions[-10:] if st.session_state.used_questions else []
                        
                        # Create a more detailed prompt with recent questions to avoid
                        chain = prompt_template | llm | StrOutputParser()
                        raw = chain.invoke({
                            "difficulty": st.session_state.difficulty,
                            "topic": st.session_state.topic,
                            "used_questions": "\n- " + "\n- ".join(recent_questions) if recent_questions else "None yet"
                        })
                        
                        # Check if this is a unique question
                        question_content = raw[:300]  # Use first 300 chars as a fingerprint
                        
                        # More aggressive uniqueness check
                        is_unique = True
                        for existing in st.session_state.question_cache:
                            # Check similarity - if more than 50% of the first 300 chars match, consider it similar
                            similarity = sum(a == b for a, b in zip(question_content.lower(), existing.lower())) / len(question_content) if len(question_content) > 0 else 0
                            if similarity > 0.5:
                                is_unique = False
                                break
                                
                        if is_unique:
                            st.session_state.question_cache.add(question_content)
                            st.session_state.used_questions.append(question_content)
                            
                            # Parse the question
                            q, opts, ans, exp = parse_question(raw)
                            
                            # Generate explanation if not provided
                            if not exp:
                                exp = generate_explanation(q, ans)
                            
                            # Verify we got a valid question
                            if q != "Error" and ans != "X" and len(opts) == 4:
                                st.session_state[key] = {
                                    "question": q,
                                    "options": opts,
                                    "answer": ans,
                                    "explanation": exp,
                                    "difficulty": st.session_state.difficulty,
                                    "is_boolean": False  # LLM always generates MCQs
                                }
                                break
                            
                # If we couldn't generate a unique question after all attempts
                if key not in st.session_state:
                    # Use a fallback question
                    st.session_state[key] = {
                        "question": f"Question about {st.session_state.topic} (fallback due to uniqueness constraints)",
                        "options": [f"{letter}. Option {letter}" for letter in ['A', 'B', 'C', 'D']],
                        "answer": "A",
                        "explanation": f"This is a fallback question because we couldn't generate a unique question about {st.session_state.topic} after {max_attempts} attempts.",
                        "difficulty": st.session_state.difficulty,
                        "is_boolean": False
                    }

        qdata = st.session_state[key]
        q, opts, ans, exp = qdata["question"], qdata["options"], qdata["answer"], qdata["explanation"]
        is_boolean = qdata.get("is_boolean", False)

    st.subheader(f"Question {st.session_state.qn + 1} of {st.session_state.total_questions}")
    if "difficulty" in qdata:
        st.caption(f"Difficulty: {qdata['difficulty']}")
    st.markdown(f"**{q}**")
    
    # Display only True/False options for boolean questions
    if is_boolean:
        selected = st.radio("Choose an option:", 
                            options=opts[:2],  # Only show first two options (True/False)
                            index=None,
                            key=f"q{st.session_state.qn}")
    else:
        selected = st.radio("Choose an option:", 
                            options=opts,
                            index=None,
                            key=f"q{st.session_state.qn}")

    if not st.session_state.answered:
        if st.button("Submit Answer", disabled=selected is None):
            user_choice = selected[0].strip().upper() if selected else "X"
            is_correct = user_choice == ans
            st.session_state.score += 1 if is_correct else 0
            st.session_state.results.append({
                "Question": q,
                "User Answer": user_choice,
                "Correct Answer": ans,
                "Is Correct": is_correct,
                "Explanation": exp,
                "Difficulty": qdata.get("difficulty", "Not Specified")
            })
            st.session_state.answered = True
            st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.qn > 0 and st.button("← Previous Question"):
                st.session_state.qn -= 1
                st.rerun()
        with col2:
            if st.session_state.qn < st.session_state.total_questions - 1:
                if st.button("Next Question →"):
                    st.session_state.qn += 1
                    st.session_state.answered = False
                    st.rerun()
            else:
                if st.button("Finish Quiz"):
                    st.session_state.qn += 1
                    st.rerun()

# Final Results
elif st.session_state.qn == st.session_state.total_questions:
    st.success("Quiz Complete!")
    st.balloons()
    
    score = st.session_state.score
    total = st.session_state.total_questions
    percent = (score / total) * 100
    
    st.subheader("Results Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Score", f"{score}/{total}")
    with col2:
        st.metric("Percentage", f"{percent:.1f}%")
    st.progress(percent / 100)
    
    # Performance feedback
    if percent >= 80:
        st.success("Excellent work! You've mastered this topic.")
    elif percent >= 60:
        st.info("Good job! You have a solid understanding.")
    elif percent >= 40:
        st.warning("Not bad! Keep practicing to improve.")
    else:
        st.error("Keep learning! Review the material and try again.")
    
    # Explanation toggle
    st.session_state.show_explanations = st.checkbox(
        "Show Explanations", 
        value=st.session_state.show_explanations,
        key="explanation_toggle"
    )
    
    # Display results
    st.subheader("Question Review")
    for i, result in enumerate(st.session_state.results):
        with st.expander(f"Question {i+1} ({result.get('Difficulty', 'N/A')})", expanded=False):
            st.markdown(f"**Question:** {result.get('Question', 'No question text')}")
            
            user_ans = result.get('User Answer', 'N/A')
            correct_ans = result.get('Correct Answer', 'N/A')
            is_correct = result.get('Is Correct', False)
            
            status = "Correct" if is_correct else "Incorrect"
            color = "green" if is_correct else "red"
            
            st.markdown(f"**Your Answer:** :{color}[{user_ans}] ({status})")
            st.markdown(f"**Correct Answer:** {correct_ans}")
            
            if st.session_state.show_explanations:
                explanation = result.get('Explanation', 'No explanation available')
                st.markdown("**Explanation:**")
                st.info(explanation)

    # Download results
    st.subheader("Download Results")
    df = pd.DataFrame(st.session_state.results)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Result CSV", 
        csv, 
        f"{st.session_state.user_name}_quiz_results.csv", 
        "text/csv"
    )
    


    if st.button("Restart Quiz"):
        # Preserve used questions to avoid repeats
        used_qs = st.session_state.used_questions.copy()
        
        # Reset session state
        for key in list(st.session_state.keys()):
            if key not in ['user_name', 'difficulty_mapping', 'used_questions']:
                del st.session_state[key]
        
        # Restore preserved values
        st.session_state.used_questions = used_qs[-50:]  # Keep recent 50 to avoid repeats
        st.session_state.quiz_started = False
        st.rerun()