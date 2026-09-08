import streamlit as st
import threading
import time
import random
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from main import load_pdf, chunk_documents, get_or_build_vectorstore, EMBEDDING_MODEL
from prompt_template import (
    get_chat_model,
    build_question_prompt,
    generate_interview_questions,
    build_answer_prompt,
    generate_all_answers,
)

READING_MESSAGES = [
    "📄 Interrogating the PDF...",
    "🔍 Sleuthing through the job description...",
    "🕵️ Investigating buzzwords...",
    "📋 Frisking the fine print...",
    "🧐 Reading between the bullet points...",
]

QUESTION_MESSAGES = [
    "🧠 Cogitating on competencies...",
    "💭 Mulling over what actually matters here...",
    "🎯 Reckoning with role requirements...",
    "❓ Conjuring tricky questions...",
    "🧩 Assembling the interviewer's arsenal...",
    "🎭 Channeling my inner intimidating interviewer...",
]

ANSWER_MESSAGES = [
    "✍️ Drafting model answers...",
    "📚 Consulting the context oracle...",
    "🤓 Overthinking sample responses...",
    "🎓 Coaching from the sidelines...",
    "💡 Connecting the dots...",
    "🔗 Cross-referencing the JD...",
]


def run_with_loader(func, messages, container, args=(), kwargs=None, interval=1.3):
    """Runs func() in the background while cycling fun status text in `container`."""
    kwargs = kwargs or {}
    result = {}

    def target():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target)
    thread.start()

    shuffled = messages[:]
    random.shuffle(shuffled)
    i = 0
    while thread.is_alive():
        container.markdown(f"#### {shuffled[i % len(shuffled)]}")
        time.sleep(interval)
        i += 1

    thread.join()
    container.empty()
    if "error" in result:
        raise result["error"]
    return result["value"]

st.title("Interview Questions Generator on Job Description")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if uploaded_file and st.button("Generate QnA", disabled=st.session_state.get("running", False)):
    st.session_state.running = True
    status = st.empty()

    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    def build_index():
        docs = load_pdf(temp_path)
        chunks = chunk_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, max_retries=5)
        vectorstore = get_or_build_vectorstore(chunks, embeddings)
        return vectorstore, embeddings

    vectorstore, embeddings = run_with_loader(build_index, READING_MESSAGES, status)

    def get_questions():
        query = "what are the qualifications and technical skills required for this role?"
        question_context = "\n\n".join(
            doc.page_content for doc, score in vectorstore.similarity_search_with_score(query, k=4)
        )
        chat_model = get_chat_model()
        questions = generate_interview_questions(
            chat_model, build_question_prompt(), question_context,
            num_categories=5, questions_per_category=5
        )
        return chat_model, questions

    chat_model, interview_questions = run_with_loader(get_questions, QUESTION_MESSAGES, status)

    def get_answers():
        question_texts = [q.question for q in interview_questions]
        question_vectors = embeddings.embed_documents(question_texts)
        qa_pairs = []
        for question_text, vector in zip(question_texts, question_vectors):
            docs_and_scores = vectorstore.similarity_search_with_score_by_vector(vector, k=3)
            context = "\n\n".join(doc.page_content for doc, score in docs_and_scores)
            qa_pairs.append((question_text, context))
        answers = generate_all_answers(chat_model, build_answer_prompt(), qa_pairs)
        return {a.index: a.answer for a in answers}

    answer_by_index = run_with_loader(get_answers, ANSWER_MESSAGES, status)

    st.balloons()  # small reward for making it through

    categories = {}
    for i, q in enumerate(interview_questions):
        categories.setdefault(q.category, []).append({
            "question": q.question,
            "answer": answer_by_index.get(i, "No answer found.")
        })

    for category, items in categories.items():
        st.subheader(category)
        for item in items:
            with st.expander(item["question"]):
                st.write(item["answer"])

    st.session_state.running = False