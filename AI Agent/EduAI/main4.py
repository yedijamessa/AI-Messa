import streamlit as st
import openai
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from fpdf import FPDF
from ics import Calendar, Event
import pytz
from PIL import Image
import os
import sys
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Import optional dependencies safely
try:
    import docx2txt
except ImportError:
    pass

try:
    from pytesseract import pytesseract
except ImportError:
    pass

try:
    from PyPDF2 import PdfReader as PyPDF2Reader
except ImportError:
    try:
        import pypdf
    except ImportError:
        pass

# ------------------ CONFIGURATION ------------------
st.set_page_config(page_title="EduAI Agent 📚", page_icon="🤖", layout="wide")
st.title("EduAI: Your Personal Learning Companion 🤖📖")

# Initialize or load session state
if 'learning_log' not in st.session_state:
    st.session_state.learning_log = []
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'generated_quiz' not in st.session_state:
    st.session_state.generated_quiz = None
if 'correct_answers' not in st.session_state:
    st.session_state.correct_answers = []
if 'quiz_difficulty' not in st.session_state:
    st.session_state.quiz_difficulty = "medium"
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'study_streak' not in st.session_state:
    st.session_state.study_streak = 0
if 'study_planner' not in st.session_state:
    st.session_state.study_planner = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'kb_files_uploaded' not in st.session_state:
    st.session_state.kb_files_uploaded = False
if 'kb_file_names' not in st.session_state:
    st.session_state.kb_file_names = []

# ------------------ SIDEBAR ------------------
st.sidebar.header("API Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# ------------------ RAG KNOWLEDGE BASE ------------------
def create_knowledge_base(files_dir="knowledge_base"):
    """Create or update the RAG knowledge base"""
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
        st.sidebar.info(f"Created knowledge base directory: {files_dir}")
    
    try:
        # Check for required packages
        required_packages = {
            "PyPDF2": "PyPDF2",
            "pypdf": "pypdf",
            "docx2txt": "docx2txt"
        }
        
        missing_packages = []
        for package_name, import_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            install_command = "pip install " + " ".join(missing_packages)
            st.sidebar.warning(f"Missing packages: {', '.join(missing_packages)}. Install with: `{install_command}`")
            
            # Try to proceed with formats we can handle
            if "pypdf" in missing_packages and "PyPDF2" in missing_packages:
                st.sidebar.warning("PDF processing disabled due to missing packages.")
            if "docx2txt" in missing_packages:
                st.sidebar.warning("DOCX processing disabled due to missing packages.")
        
        # Load documents from directory
        documents = []
        
        # Always try loading text files
        try:
            text_loader = DirectoryLoader(files_dir, glob="**/*.txt", loader_cls=TextLoader)
            txt_docs = text_loader.load()
            documents.extend(txt_docs)
            st.sidebar.success(f"Loaded {len(txt_docs)} text files.")
        except Exception as e:
            st.sidebar.warning(f"Error loading text files: {e}")
        
        # Try loading PDF files if dependencies available
        if "pypdf" not in missing_packages or "PyPDF2" not in missing_packages:
            try:
                pdf_loader = DirectoryLoader(files_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                st.sidebar.success(f"Loaded {len(pdf_docs)} PDF files.")
            except Exception as e:
                st.sidebar.warning(f"Error loading PDF files: {e}")
        
        # Try loading DOCX files if dependencies available
        if "docx2txt" not in missing_packages:
            try:
                docx_loader = DirectoryLoader(files_dir, glob="**/*.docx", loader_cls=Docx2txtLoader)
                docx_docs = docx_loader.load()
                documents.extend(docx_docs)
                st.sidebar.success(f"Loaded {len(docx_docs)} DOCX files.")
            except Exception as e:
                st.sidebar.warning(f"Error loading DOCX files: {e}")
        
        if not documents:
            st.sidebar.info("No documents found in knowledge base. Please add some files to enhance responses.")
            return None
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(texts, embeddings)
        
        st.sidebar.success(f"Knowledge base updated with {len(documents)} documents and {len(texts)} chunks.")
        return vector_store
    
    except Exception as e:
        st.sidebar.error(f"Error creating knowledge base: {e}")
        return None

# ------------------ RAG FUNCTIONALITY ------------------
def get_rag_response(query, vector_store, learning_context=None):
    """Get a response using RAG."""
    if vector_store is None:
        return None
    
    try:
        # Create a retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Create a chain
        llm = OpenAI(temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        
        # Add learning context if available
        enhanced_query = query
        if learning_context:
            enhanced_query = f"Context from student's learning: {learning_context}\n\nQuery: {query}"
        
        # Get response
        response = qa_chain.run(enhanced_query)
        return response
    
    except Exception as e:
        st.error(f"Error in RAG response: {e}")
        return None

# ------------------ KNOWLEDGE BASE MANAGEMENT ------------------
st.sidebar.header("Knowledge Base")
kb_tab1, kb_tab2 = st.sidebar.tabs(["📚 Upload Files", "🔄 Update KB"])

with kb_tab1:
    # Display already uploaded files
    if st.session_state.kb_files_uploaded and st.session_state.kb_file_names:
        st.sidebar.success(f"Files in knowledge base: {', '.join(st.session_state.kb_file_names)}")
    
    kb_files = st.file_uploader("Upload files to your knowledge base", 
                               type=["txt", "pdf", "docx"], 
                               accept_multiple_files=True)
    
    if kb_files and st.sidebar.button("Add to Knowledge Base"):
        # Create knowledge base directory if it doesn't exist
        if not os.path.exists("knowledge_base"):
            os.makedirs("knowledge_base")
        
        # Save uploaded files
        added_files = 0
        new_file_names = []
        for file in kb_files:
            try:
                file_path = os.path.join("knowledge_base", file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                added_files += 1
                new_file_names.append(file.name)
            except Exception as e:
                st.sidebar.error(f"Error saving {file.name}: {e}")
        
        # Update session state
        st.session_state.kb_files_uploaded = True
        st.session_state.kb_file_names.extend(new_file_names)
        
        st.sidebar.success(f"Added {added_files} files to knowledge base.")
        
        # Install required packages notice
        required_packages = []
        for file in kb_files:
            if file.name.endswith(".pdf") and not any(p in sys.modules for p in ["PyPDF2", "pypdf"]):
                required_packages.append("pypdf")
            elif file.name.endswith(".docx") and "docx2txt" not in sys.modules:
                required_packages.append("docx2txt")
        
        if required_packages:
            st.sidebar.warning(f"Some files may require additional packages. Run: pip install {' '.join(required_packages)}")
        
        # Update the knowledge base
        with st.sidebar:
            kb_status = st.empty()
            kb_status.info("Building knowledge base...")
            st.session_state.knowledge_base = create_knowledge_base()
            if st.session_state.knowledge_base is not None:
                kb_status.success("Knowledge base built successfully!")
            else:
                kb_status.error("Failed to build knowledge base. Check for errors above.")

with kb_tab2:
    if st.sidebar.button("Refresh Knowledge Base"):
        with st.sidebar:
            refresh_status = st.empty()
            refresh_status.info("Refreshing knowledge base...")
            st.session_state.knowledge_base = create_knowledge_base()
            refresh_status.success("Knowledge base refreshed!")

# ------------------ MAIN TABS ------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📚 Learning Log", "🤖 AI Suggestions", "📝 Quiz Generator", "📊 Progress Tracker", "🌐 Web Search", "📅 Study Planner", "💬 Ask the AI"])

# ------------------ LEARNING LOG ------------------
with tab1:
    st.subheader("📝 What Did You Learn Today?")

    student_name = st.text_input("Enter your name:", "Student")
    subject = st.text_input("Subject/Topic:", "Mathematics")
    learned_content = st.text_area("Describe what you've learned:", "E.g., I learned about algebraic expressions.")

    if st.button("Log Learning"):
        st.session_state.learning_log.append({
            "Date": datetime.date.today().strftime("%Y-%m-%d"),
            "Student": student_name,
            "Subject": subject,
            "Learned": learned_content
        })
        st.session_state.points += 10
        st.success("Learning logged successfully! +10 points 🎉")

    if st.session_state.learning_log:
        df_log = pd.DataFrame(st.session_state.learning_log)
        st.subheader("📊 Your Learning Log")
        st.dataframe(df_log, use_container_width=True)

        if st.button("Download Learning Log as CSV"):
            df_log.to_csv("learning_log.csv", index=False)
            st.success("Learning log exported successfully! Check your downloads.")

# ------------------ AI RECOMMENDATION ------------------
with tab2:
    st.subheader("💡 What Should I Learn Next?")
    next_question = st.text_input("Ask the AI for learning suggestions:", "What should I learn next in this subject?")

    if st.button("Get Suggestion"):
        if learned_content:
            try:
                # Get recent learning history for context
                recent_learning = ""
                if st.session_state.learning_log:
                    last_3_entries = st.session_state.learning_log[-3:]
                    for entry in last_3_entries:
                        recent_learning += f"Subject: {entry['Subject']}, Learned: {entry['Learned']}\n"
                
                # Get quiz history for context
                quiz_context = ""
                if st.session_state.quiz_history:
                    # Group quiz history by subject
                    subject_history = {}
                    for item in st.session_state.quiz_history:
                        if item['subject'] not in subject_history:
                            subject_history[item['subject']] = {"correct": 0, "total": 0, "topics": set()}
                        
                        subject_history[item['subject']]["total"] += 1
                        if item["correct"]:
                            subject_history[item['subject']]["correct"] += 1
                        subject_history[item['subject']]["topics"].add(item["topic"])
                    
                    # Add context about the current subject
                    if subject in subject_history:
                        correct = subject_history[subject]["correct"]
                        total = subject_history[subject]["total"]
                        accuracy = (correct / total) * 100 if total > 0 else 0
                        topics = ", ".join(list(subject_history[subject]["topics"]))
                        
                        quiz_context = f"Quiz performance in {subject}: {correct}/{total} correct answers ({accuracy:.1f}%). Topics tested: {topics}."
                
                # Try RAG first
                rag_response = None
                if st.session_state.knowledge_base:
                    rag_prompt = f"Student is studying {subject} and learned: {learned_content}. The student wants to know what to study next."
                    rag_response = get_rag_response(rag_prompt, st.session_state.knowledge_base, recent_learning)
                
                # If RAG fails or isn't available, use regular OpenAI API
                if not rag_response:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a helpful educational assistant. Use the following context to provide relevant suggestions: {quiz_context}"},
                            {"role": "user", "content": f"I am a student studying {subject}. Today I learned: {learned_content}. What should I study next to improve my knowledge? Consider my quiz performance in your recommendation."}
                        ],
                        max_tokens=150
                    )
                    suggestion = response.choices[0].message.content.strip()
                else:
                    suggestion = rag_response

                st.markdown(f"**📚 Suggested Next Topics:**\n{suggestion}")

                # Resource Recommendations
                resource_prompt = f"Suggest 3 online resources (videos, books, or articles) to deepen my understanding of {subject}."
                
                # Try RAG for resources too
                resource_rag = None
                if st.session_state.knowledge_base:
                    resource_rag = get_rag_response(resource_prompt, st.session_state.knowledge_base, learned_content)
                
                if not resource_rag:
                    resource_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a resourceful educational assistant."},
                            {"role": "user", "content": resource_prompt}
                        ],
                        max_tokens=200
                    )
                    resources = resource_response.choices[0].message.content.strip()
                else:
                    resources = resource_rag
                
                st.markdown(f"**🌐 Recommended Resources:**\n{resources}")

            except Exception as e:
                st.error(f"Error fetching AI suggestion: {e}")
        else:
            st.warning("Please log what you've learned first before asking for suggestions.")

# ------------------ QUIZ GENERATOR ------------------
with tab3:
    st.subheader("📝 Quiz & Practice Generator")

    if st.session_state.learning_log:
        # Extract unique subjects from Learning Log
        logged_subjects = list(set([log['Subject'] for log in st.session_state.learning_log]))

        # Step 1: Select or Enter Subject
        subject_mode = st.radio("Choose Subject Mode:", ["📚 Select from Logged Subjects", "💡 Enter Custom Subject"])

        if subject_mode == "📚 Select from Logged Subjects":
            selected_subject = st.selectbox("Select a logged subject:", options=logged_subjects)
        else:
            selected_subject = st.text_input("Enter a custom subject:")

        # Step 2: Choose Topic
        if selected_subject:
            # Filter "Learned" topics based on selected subject
            logged_topics = [log['Learned'] for log in st.session_state.learning_log if log['Subject'] == selected_subject]

            topic_mode = st.radio("Choose Topic Mode:", ["📚 Select from Logged Topics", "💡 Enter Custom Topic"])

            if topic_mode == "📚 Select from Logged Topics" and logged_topics:
                selected_topic = st.selectbox("Select a logged topic for the quiz:", options=logged_topics)
            elif topic_mode == "💡 Enter Custom Topic":
                selected_topic = st.text_input("Enter a custom topic for the quiz:")
            else:
                selected_topic = None

            # Initialize session state variables
            if 'current_question_index' not in st.session_state:
                st.session_state.current_question_index = 0
            if 'quiz_questions' not in st.session_state:
                st.session_state.quiz_questions = []
            if 'quiz_answers' not in st.session_state:
                st.session_state.quiz_answers = []
            if 'quiz_feedback' not in st.session_state:
                st.session_state.quiz_feedback = ""
            if 'quiz_score' not in st.session_state:
                st.session_state.quiz_score = 0

            # Display current difficulty level
            st.write(f"**Current Difficulty Level:** {st.session_state.quiz_difficulty.capitalize()}")
            
            # Manual difficulty selection option
            manual_difficulty = st.checkbox("Manually set difficulty level")
            if manual_difficulty:
                st.session_state.quiz_difficulty = st.radio(
                    "Select difficulty level:",
                    ["easy", "medium", "hard"],
                    index=["easy", "medium", "hard"].index(st.session_state.quiz_difficulty)
                )
            
            # Generate Quiz Button
            if st.button("Generate Quiz"):
                if not selected_topic:
                    st.warning("Please select or enter a topic.")
                else:
                    # Adjust prompt based on difficulty level
                    difficulty_instructions = {
                        "easy": "Use simple concepts and straightforward questions. Focus on fundamental knowledge and basic recall.",
                        "medium": "Include moderate complexity. Mix recall with some application and analysis questions.",
                        "hard": "Create challenging questions that require deeper understanding, analysis, and application of concepts. Include complex scenarios and edge cases."
                    }
                    
                    # Get recent quiz performance to provide to the AI
                    recent_performance = ""
                    if st.session_state.quiz_history:
                        last_5_quizzes = st.session_state.quiz_history[-5:]
                        correct_count = sum(1 for q in last_5_quizzes if q["correct"])
                        total = len(last_5_quizzes)
                        recent_performance = f"Recent quiz performance: {correct_count}/{total} correct answers."
                    
                    quiz_prompt = f"Create 5 multiple-choice questions about {selected_subject} focusing on '{selected_topic}' at {st.session_state.quiz_difficulty.upper()} difficulty level.\n\n{difficulty_instructions[st.session_state.quiz_difficulty]}\n\n{recent_performance}\n\nEach question should have four options (A/B/C/D) and include the correct answer with a reason. Format as:\nQuestion: <question>\nA) <option1>\nB) <option2>\nC) <option3>\nD) <option4>\nAnswer: <correct option letter>) <correct option>\nReason: <explanation>"
                    
                    # Try RAG first for quiz generation
                    rag_quiz = None
                    if st.session_state.knowledge_base:
                        rag_quiz = get_rag_response(quiz_prompt, st.session_state.knowledge_base)
                    
                    if not rag_quiz:
                        quiz_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a quiz generator."},
                                {"role": "user", "content": quiz_prompt}
                            ],
                            max_tokens=500
                        )
                        quiz_content = quiz_response.choices[0].message.content.strip().split('\n\n')
                    else:
                        quiz_content = rag_quiz.strip().split('\n\n')
                    
                    # Parse and store quiz questions and answers
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = []

                    for q in quiz_content:
                        try:
                            # More robust parsing to handle multiple occurrences of "Answer:"
                            if "Answer:" in q:
                                parts = q.split("Answer:", 1)  # Split only at the first occurrence
                                question_part = parts[0].strip()
                                answer_section = parts[1].strip()
                                
                                # Handle various formats for reason
                                if "Reason:" in answer_section:
                                    answer_parts = answer_section.split("Reason:", 1)
                                    answer_text = answer_parts[0].strip()
                                    reason_text = answer_parts[1].strip()
                                else:
                                    answer_text = answer_section
                                    reason_text = "No reason provided."
                                    
                                st.session_state.quiz_questions.append(question_part)
                                st.session_state.quiz_answers.append((answer_text, reason_text))
                        except Exception as e:
                            st.warning(f"Couldn't parse a quiz question properly. Skipping it. Error: {str(e)}")
                            continue

                    # Only proceed if we have at least one valid question
                    if st.session_state.quiz_questions:
                        st.session_state.current_question_index = 0
                        st.session_state.quiz_feedback = ""
                        st.session_state.quiz_score = 0  # Reset score for new quiz
                    else:
                        st.error("Couldn't generate valid quiz questions. Please try again.")

        # Display Current Question
        if st.session_state.quiz_questions:
            q_index = st.session_state.current_question_index
            if q_index < len(st.session_state.quiz_questions):
                question_text = st.session_state.quiz_questions[q_index]
                correct_answer, reason = st.session_state.quiz_answers[q_index]

                # Display Question and Options
                question_lines = question_text.split('\n')
                for line in question_lines:
                    st.markdown(line)

                # User Input
                user_answer = st.text_input("Answer (A/B/C/D):", key=f"answer_{q_index}")

                # Submit Answer
                if st.button("Submit Answer", key=f"submit_{q_index}"):
                    user_ans = user_answer.strip().upper()
                    correct_ans_letter = correct_answer.split(')')[0].strip().upper()

                    # Track answer correctness
                    is_correct = user_ans == correct_ans_letter
                    
                    # Add to quiz history
                    st.session_state.quiz_history.append({
                        "subject": selected_subject,
                        "topic": selected_topic,
                        "question": question_text,
                        "user_answer": user_ans,
                        "correct_answer": correct_ans_letter,
                        "correct": is_correct,
                        "difficulty": st.session_state.quiz_difficulty,
                        "date": datetime.date.today().strftime("%Y-%m-%d")
                    })
                    
                    # Adjust difficulty based on answer correctness
                    if is_correct:
                        st.session_state.quiz_feedback = f"✅ Right! (+10 points)\nReason: {reason}"
                        st.session_state.quiz_score += 10
                        
                        # Increase difficulty if answered correctly
                        difficulty_levels = ["easy", "medium", "hard"]
                        current_index = difficulty_levels.index(st.session_state.quiz_difficulty)
                        if current_index < len(difficulty_levels) - 1:
                            # Only increase if we've had 2 correct answers at this level
                            recent_correct = [q for q in st.session_state.quiz_history[-3:] 
                                             if q["correct"] and q["difficulty"] == st.session_state.quiz_difficulty]
                            if len(recent_correct) >= 2 and not manual_difficulty:
                                st.session_state.quiz_difficulty = difficulty_levels[current_index + 1]
                                st.session_state.quiz_feedback += f"\n\n📈 Difficulty increased to {st.session_state.quiz_difficulty.capitalize()}!"
                    else:
                        st.session_state.quiz_feedback = f"❌ Wrong! (-5 points)\nCorrect Answer: {correct_answer}\nReason: {reason}"
                        # Don't let score go below zero
                        st.session_state.quiz_score = max(0, st.session_state.quiz_score - 5)
                        
                        # Decrease difficulty if answered incorrectly
                        difficulty_levels = ["easy", "medium", "hard"]
                        current_index = difficulty_levels.index(st.session_state.quiz_difficulty)
                        if current_index > 0:
                            # Only decrease if we've had 2 incorrect answers at this level
                            recent_incorrect = [q for q in st.session_state.quiz_history[-3:] 
                                               if not q["correct"] and q["difficulty"] == st.session_state.quiz_difficulty]
                            if len(recent_incorrect) >= 2 and not manual_difficulty:
                                st.session_state.quiz_difficulty = difficulty_levels[current_index - 1]
                                st.session_state.quiz_feedback += f"\n\n📉 Difficulty decreased to {st.session_state.quiz_difficulty.capitalize()}."

                # Display Feedback
                if st.session_state.quiz_feedback:
                    st.markdown(st.session_state.quiz_feedback)
                    if st.button("Next Question", key=f"next_{q_index}"):
                        st.session_state.current_question_index += 1
                        st.session_state.quiz_feedback = ""

            else:
                # Quiz Completion
                st.success("🎉 You've completed the quiz!")
                st.markdown(f"🏆 **Total Quiz Score:** {st.session_state.quiz_score}")
                if st.button("Start New Quiz"):
                    st.session_state.current_question_index = 0
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_feedback = ""
                    st.session_state.quiz_score = 0
    else:
        st.info("Please log your learning topics to generate a quiz.")

# ------------------ PROGRESS TRACKER ------------------
with tab4:
    st.subheader("📊 Your Progress Tracker")

    if st.session_state.learning_log or st.session_state.quiz_history:
        # Create tabs for different progress views
        progress_tab1, progress_tab2, progress_tab3 = st.tabs(["📚 Learning Overview", "📝 Quiz Performance", "📈 Points History"])
        
        with progress_tab1:
            if st.session_state.learning_log:
                df_log = pd.DataFrame(st.session_state.learning_log)
                
                # Show top subjects studied
                st.write("### Topics Studied Frequency")
                topic_count = df_log['Subject'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                topic_count.plot(kind='bar', ax=ax, color='skyblue')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show learning activity over time
                st.write("### Learning Activity Timeline")
                df_log['Date'] = pd.to_datetime(df_log['Date'])
                timeline_data = df_log.groupby(df_log['Date'].dt.strftime('%Y-%m-%d')).size().reset_index(name='Entries')
                timeline_data['Date'] = pd.to_datetime(timeline_data['Date'])
                timeline_data = timeline_data.sort_values('Date')
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(timeline_data['Date'], timeline_data['Entries'], marker='o', linestyle='-', color='green')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Number of Entries')
                ax2.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("Log your learnings to see your study pattern.")
                
        with progress_tab2:
            if st.session_state.quiz_history:
                # Create DataFrame from quiz history
                df_quiz = pd.DataFrame(st.session_state.quiz_history)
                
                # Show quiz performance by subject
                st.write("### Quiz Performance by Subject")
                subject_performance = df_quiz.groupby('subject').agg(
                    total=('correct', 'count'),
                    correct=('correct', 'sum')
                ).reset_index()
                subject_performance['accuracy'] = (subject_performance['correct'] / subject_performance['total'] * 100).round(1)
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.bar(subject_performance['subject'], subject_performance['accuracy'], color='purple')
                ax3.set_xlabel('Subject')
                ax3.set_ylabel('Accuracy (%)')
                ax3.set_ylim(0, 100)
                for i, v in enumerate(subject_performance['accuracy']):
                    ax3.text(i, v + 2, f"{v}%", ha='center')
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Show difficulty progression
                st.write("### Difficulty Progression")
                df_quiz['date'] = pd.to_datetime(df_quiz['date'])
                df_quiz_sorted = df_quiz.sort_values('date')
                difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3}
                df_quiz_sorted['difficulty_numeric'] = df_quiz_sorted['difficulty'].map(difficulty_map)
                
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.plot(range(len(df_quiz_sorted)), df_quiz_sorted['difficulty_numeric'], marker='o', linestyle='-', color='orange')
                ax4.set_xlabel('Question Number')
                ax4.set_ylabel('Difficulty Level')
                ax4.set_yticks([1, 2, 3])
                ax4.set_yticklabels(['Easy', 'Medium', 'Hard'])
                plt.tight_layout()
                st.pyplot(fig4)
                
                # Show performance by difficulty
                st.write("### Performance by Difficulty Level")
                difficulty_performance = df_quiz.groupby('difficulty').agg(
                    total=('correct', 'count'),
                    correct=('correct', 'sum')
                ).reset_index()
                difficulty_performance['accuracy'] = (difficulty_performance['correct'] / difficulty_performance['total'] * 100).round(1)
                
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                
                # Create a safe version that doesn't rely on reindexing with loc
                # Instead, we'll manually sort and handle missing difficulties
                difficulty_order = {'easy': 0, 'medium': 1, 'hard': 2}
                
                # Sort by our custom order
                difficulty_performance['sort_order'] = difficulty_performance['difficulty'].map(difficulty_order)
                difficulty_performance = difficulty_performance.sort_values('sort_order')
                
                # Define colors based on what difficulties are present
                bar_colors = []
                for diff in difficulty_performance['difficulty']:
                    if diff == 'easy':
                        bar_colors.append('lightgreen')
                    elif diff == 'medium':
                        bar_colors.append('yellow')
                    elif diff == 'hard':
                        bar_colors.append('salmon')
                
                ax5.bar(difficulty_performance['difficulty'], difficulty_performance['accuracy'], color=bar_colors)
                ax5.set_xlabel('Difficulty')
                ax5.set_ylabel('Accuracy (%)')
                ax5.set_ylim(0, 100)
                for i, v in enumerate(difficulty_performance['accuracy']):
                    ax5.text(i, v + 2, f"{v}%", ha='center')
                plt.tight_layout()
                st.pyplot(fig5)
            else:
                st.info("Take some quizzes to see your performance analytics.")
                
        with progress_tab3:
            # Calculate total points
            learning_points = st.session_state.points
            quiz_points = st.session_state.quiz_score
            total_points = learning_points + quiz_points
            
            # Display points breakdown
            col1, col2, col3 = st.columns(3)
            col1.metric("Learning Log Points", learning_points)
            col2.metric("Quiz Points", quiz_points)
            col3.metric("Total Points", total_points)
            
            # Create a visual breakdown
            st.write("### Points Breakdown")
            
            # Ensure we only use positive values for the pie chart
            learning_points_display = max(0, learning_points)
            quiz_points_display = max(0, quiz_points)
            
            # Only create pie chart if we have positive values
            if learning_points_display > 0 or quiz_points_display > 0:
                fig6, ax6 = plt.subplots(figsize=(8, 8))
                
                # Create labels and values arrays, only including positive values
                labels = []
                values = []
                
                if learning_points_display > 0:
                    labels.append('Learning Log')
                    values.append(learning_points_display)
                    
                if quiz_points_display > 0:
                    labels.append('Quizzes')
                    values.append(quiz_points_display)
                
                if labels and values:  # Only create pie if we have data
                    ax6.pie(
                        values,
                        labels=labels,
                        autopct='%1.1f%%',
                        colors=['#4CAF50', '#3F51B5'],
                        startangle=90,
                        shadow=True
                    )
                    ax6.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig6)
                else:
                    st.info("No positive points to display in the breakdown chart yet.")
            else:
                st.info("Start earning points to see your breakdown chart!")
            
            # Points recommendations
            if total_points < 100:
                st.info("💡 **Tip:** Log more of your learning activities and take quizzes to earn more points!")
            elif total_points < 500:
                st.success("🌟 Good progress! Keep logging your learning and challenging yourself with quizzes.")
            else:
                st.success("🏆 Outstanding effort! You're making excellent progress in your learning journey.")
    else:
        st.info("Log your learnings and take quizzes to see your progress here.")

# ------------------ WEB SEARCH TAB ------------------
with tab5:
    st.subheader("🌐 Search the Web")
    search_query = st.text_input("Enter your search query:", "Latest AI research")

    if st.button("Search the Web"):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(search_query, region='wt-wt', safesearch='moderate', timelimit='y')
                st.write(f"**🔍 Search Results for:** {search_query}")

                for result in results:
                    title = result.get("title", "No Title")
                    link = result.get("href", "No Link")
                    snippet = result.get("body", "")
                    st.markdown(f"**[{title}]({link})**\n\n{snippet}\n\n---")

        except Exception as e:
            st.error(f"Error during web search: {e}")

# ------------------ STUDY PLANNER ------------------
with tab6:
    st.subheader("📅 Study Planner")

    start_date = st.date_input("Select start date:", datetime.date.today())
    end_date = st.date_input("Select end date:", datetime.date.today())
    planner_topic = st.text_input("What will you study?")
    daily_hours = st.number_input("How many hours can you study per day?", min_value=1, max_value=12, value=2)

    if st.button("Add to Planner"):
        if start_date > end_date:
            st.error("End date must be after start date.")
        else:
            total_days = (end_date - start_date).days + 1
            study_plan = []
            
            # Try to use RAG for study plan
            use_rag = st.session_state.knowledge_base is not None
            
            for i in range(total_days):
                day = start_date + datetime.timedelta(days=i)
                suggestion_prompt = f"Suggest a detailed {daily_hours}-hour study plan for {planner_topic} on {day.strftime('%Y-%m-%d')}"
                
                if use_rag:
                    # Get recent learning as context
                    recent_learning = ""
                    if st.session_state.learning_log:
                        last_3_entries = st.session_state.learning_log[-3:]
                        for entry in last_3_entries:
                            recent_learning += f"Subject: {entry['Subject']}, Learned: {entry['Learned']}\n"
                    
                    rag_suggestion = get_rag_response(suggestion_prompt, st.session_state.knowledge_base, recent_learning)
                    if rag_suggestion:
                        suggestion = rag_suggestion
                    else:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an educational assistant."},
                                {"role": "user", "content": suggestion_prompt}
                            ],
                            max_tokens=300
                        )
                        suggestion = response.choices[0].message.content.strip()
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an educational assistant."},
                            {"role": "user", "content": suggestion_prompt}
                        ],
                        max_tokens=300
                    )
                    suggestion = response.choices[0].message.content.strip()
                
                study_plan.append({
                    "Date": day.strftime("%Y-%m-%d"),
                    "Topic": planner_topic,
                    "Suggested Study": suggestion,
                    "Hours": daily_hours,
                    "Progress": 0
                })

            # Replace existing plan instead of appending
            st.session_state.study_planner = study_plan
            st.success("Study plan updated!")

    # 💡 AI Planner Modification that updates the plan directly
    st.subheader("💡 AI Planner - Modify Study Plan")
    modify_prompt = st.text_area("Ask AI to modify your study plan (e.g., 'Reduce study time', 'Add more exercises'):") 

    if st.button("Apply AI Modification"):
        if modify_prompt and st.session_state.study_planner:
            for i, plan in enumerate(st.session_state.study_planner):
                modify_plan_prompt = f"Here is the current study plan:\n{plan['Suggested Study']}\n\n{modify_prompt}\nPlease update the plan accordingly."
                
                # Try RAG first for plan modification
                if st.session_state.knowledge_base:
                    rag_modified_plan = get_rag_response(modify_plan_prompt, st.session_state.knowledge_base)
                    if rag_modified_plan:
                        updated_suggestion = rag_modified_plan
                    else:
                        ai_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an educational assistant."},
                                {"role": "user", "content": modify_plan_prompt}
                            ],
                            max_tokens=300
                        )
                        updated_suggestion = ai_response.choices[0].message.content.strip()
                else:
                    ai_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an educational assistant."},
                            {"role": "user", "content": modify_plan_prompt}
                        ],
                        max_tokens=300
                    )
                    updated_suggestion = ai_response.choices[0].message.content.strip()

                # Apply AI modification to the specific plan
                st.session_state.study_planner[i]['Suggested Study'] = updated_suggestion

            st.success("Study plan modified successfully!")

    # Display Study Plans and Add Progress Tracking
    if st.session_state.study_planner:
        st.write("### Your Study Plans")
        for plan in st.session_state.study_planner:
            progress = st.slider(f"Progress for {plan['Date']} - {plan['Topic']}", 0, 100, plan['Progress'], key=f"progress_{plan['Date']}")
            plan['Progress'] = progress
            st.write(f"📅 {plan['Date']} | 🕒 {plan['Hours']} hrs | 📖 {plan['Topic']} | 💡 {plan['Suggested Study']} | ✅ Progress: {plan['Progress']}%")

    # 📅 Export to iCal/Google Calendar
    if st.button("Export to iCal/Google Calendar"):
        c = Calendar()
        timezone = pytz.timezone("UTC")
        for plan in st.session_state.study_planner:
            e = Event()
            e.name = plan['Topic']
            e.begin = f"{plan['Date']} 09:00:00"
            e.end = f"{plan['Date']} {9 + plan['Hours']}:00:00"
            e.description = plan['Suggested Study']
            e.uid = f"{plan['Date']}-{plan['Topic']}"
            c.events.add(e)
        with open("study_plan.ics", "w") as f:
            f.writelines(c)
        st.success("Study plan exported! You can now import it into Google Calendar or iCal.")

# ------------------ ASK THE AI TAB ------------------
with tab7:
    st.subheader("💬 Ask the AI Anything")
    user_prompt = st.text_area("Enter your question or request for the AI:")

    # Keep track of uploaded files in session state
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""

    uploaded_file = st.file_uploader("Upload an image (JPG, PNG), PDF, or DOCX file for homework help:", 
                                    type=["jpg", "jpeg", "png", "pdf", "docx"])
    
    # Process the uploaded file if it's new
    if uploaded_file is not None and (st.session_state.last_uploaded_file is None or 
                                     uploaded_file.name != getattr(st.session_state.last_uploaded_file, 'name', None)):
        st.session_state.last_uploaded_file = uploaded_file
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension in ["jpg", "jpeg", "png"]:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Check if pytesseract is available
                try:
                    import pytesseract
                    try:
                        st.session_state.extracted_text = pytesseract.image_to_string(image)
                        if st.session_state.extracted_text:
                            st.success("Text extracted from image successfully!")
                            with st.expander("View extracted text"):
                                st.text(st.session_state.extracted_text)
                        else:
                            st.warning("No text detected in the image. Please ensure the image contains readable text.")
                    except Exception as e:
                        st.error(f"Error during OCR: {e}")
                        st.warning("Make sure Tesseract OCR is installed on your system. For installation instructions, visit: https://github.com/tesseract-ocr/tesseract")
                except ImportError:
                    st.error("pytesseract is not installed. Install with: pip install pytesseract")
                    st.warning("You also need to install Tesseract OCR on your system: https://github.com/tesseract-ocr/tesseract")
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        elif file_extension == "pdf":
            try:
                # Try different PDF libraries
                pdf_text = ""
                try:
                    from PyPDF2 import PdfReader
                    pdf_reader = PdfReader(uploaded_file)
                    pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                except ImportError:
                    try:
                        import pypdf
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    except ImportError:
                        st.error("No PDF library found. Install PyPDF2 or pypdf with: pip install pypdf")
                
                if pdf_text:
                    st.session_state.extracted_text = pdf_text
                    st.success("Text extracted from PDF successfully!")
                    with st.expander("View extracted text"):
                        st.text(pdf_text[:500] + ("..." if len(pdf_text) > 500 else ""))
                else:
                    st.warning("No text could be extracted from the PDF. It might be a scanned document or protected.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        
        elif file_extension == "docx":
            try:
                try:
                    import docx2txt
                    docx_text = docx2txt.process(uploaded_file)
                    if docx_text:
                        st.session_state.extracted_text = docx_text
                        st.success("Text extracted from DOCX successfully!")
                        with st.expander("View extracted text"):
                            st.text(docx_text[:500] + ("..." if len(docx_text) > 500 else ""))
                    else:
                        st.warning("The DOCX file appears to be empty.")
                except ImportError:
                    st.error("docx2txt is not installed. Install with: pip install docx2txt")
            except Exception as e:
                st.error(f"Error processing DOCX: {e}")
    
    # Display already extracted text if available
    elif st.session_state.extracted_text and not uploaded_file:
        st.info("Using previously extracted text.")
        with st.expander("View extracted text"):
            st.text(st.session_state.extracted_text[:500] + ("..." if len(st.session_state.extracted_text) > 500 else ""))
    
    chat_mode = st.radio("Choose mode:", ["Standard", "RAG-enhanced"])
    
    if st.button("Ask AI"):
        if user_prompt or st.session_state.extracted_text:
            with st.spinner("Thinking..."):
                try:
                    # Prepare the query with the extracted text as context
                    if user_prompt and st.session_state.extracted_text:
                        query_content = f"{user_prompt}\n\nContext from uploaded document:\n{st.session_state.extracted_text}"
                    elif st.session_state.extracted_text:
                        query_content = f"Please analyze this content and provide insights:\n\n{st.session_state.extracted_text}"
                    else:
                        query_content = user_prompt
                    
                    # Get learning context from log
                    learning_context = ""
                    if st.session_state.learning_log:
                        last_5_entries = st.session_state.learning_log[-5:]
                        for entry in last_5_entries:
                            learning_context += f"Subject: {entry['Subject']}, Learned: {entry['Learned']}\n"
                    
                    if chat_mode == "RAG-enhanced" and st.session_state.knowledge_base:
                        # Use RAG for response
                        rag_answer = get_rag_response(query_content, st.session_state.knowledge_base, learning_context)
                        
                        if rag_answer:
                            ai_answer = rag_answer
                            st.markdown(f"**🤖 AI Response (RAG-enhanced):**\n{ai_answer}")
                        else:
                            # Fallback to standard mode
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an educational assistant that helps students with their homework and study materials."},
                                    {"role": "user", "content": query_content}
                                ],
                                max_tokens=800
                            )
                            ai_answer = response.choices[0].message.content.strip()
                            st.markdown(f"**🤖 AI Response (Standard):**\n{ai_answer}")
                    else:
                        # Use standard mode
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an educational assistant that helps students with their homework and study materials."},
                                {"role": "user", "content": query_content}
                            ],
                            max_tokens=800
                        )
                        ai_answer = response.choices[0].message.content.strip()
                        st.markdown(f"**🤖 AI Response:**\n{ai_answer}")

                except Exception as e:
                    st.error(f"Error during AI interaction: {e}")
        else:
            st.warning("Please enter a question or upload a file.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("💡 *Empowering students to take control of their learning journey.*")