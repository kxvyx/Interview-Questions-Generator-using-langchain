import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS, DistanceStrategy
from prompt_template import (
    get_chat_model,
    build_question_prompt,
    generate_interview_questions,
    build_answer_prompt,
    generate_all_answers,
)

load_dotenv()
EMBEDDING_MODEL = "models/gemini-embedding-001"

def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return list(loader.lazy_load())

def chunk_documents(docs, chunk_size=600, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", " "]
    )
    return splitter.split_documents(docs)

def get_or_build_vectorstore(chunks, embeddings, index_path="faiss_index"):
    meta_path = f"{index_path}_meta.json"
    current_meta = {"embedding_model": EMBEDDING_MODEL, "num_chunks": len(chunks)}

    if os.path.exists(index_path) and os.path.exists(meta_path):
        if json.load(open(meta_path)) == current_meta:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
    vectorstore.save_local(index_path)
    json.dump(current_meta, open(meta_path, "w"))
    return vectorstore

def main():
    docs = load_pdf("Deloitte_JD.pdf")
    chunks = chunk_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, max_retries=5)
    vectorstore = get_or_build_vectorstore(chunks, embeddings)

    # Context for question generation
    query = "what are the qualifications and technical skills required for this role?"
    question_context = "\n\n".join(
        doc.page_content for doc, score in vectorstore.similarity_search_with_score(query, k=4)
    )

    # 1 API call: generate all questions
    chat_model = get_chat_model()
    interview_questions = generate_interview_questions(
        chat_model, build_question_prompt(), question_context,
        num_categories=5, questions_per_category=5
    )

    # 1 API call: batch-embed all questions
    question_texts = [q.question for q in interview_questions]
    question_vectors = embeddings.embed_documents(question_texts)

    # Local FAISS lookups (free, no API calls) — one per question, per objective 1
    qa_pairs = []
    for question_text, vector in zip(question_texts, question_vectors):
        docs_and_scores = vectorstore.similarity_search_with_score_by_vector(vector, k=3)
        context = "\n\n".join(doc.page_content for doc, score in docs_and_scores)
        qa_pairs.append((question_text, context))

    # 1 API call: generate all answers, matched back by index
    answers = generate_all_answers(chat_model, build_answer_prompt(), qa_pairs)
    answer_by_index = {a.index: a.answer for a in answers}

    QnA_pairs = []
    for i, q in enumerate(interview_questions):
        QnA_pairs.append({
            "category": q.category,
            "question": q.question,
            "answer": answer_by_index.get(i, "No answer found.")
        })

    for pair in QnA_pairs:
        print(f"Category: {pair['category']}")
        print(f"Question: {pair['question']}")
        print(f"Answer: {pair['answer']}\n")

    return QnA_pairs

if __name__ == "__main__":
    main()