import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, DistanceStrategy
from prompt_template import get_chat_model, build_prompt, generate_interview_questions

load_dotenv()

def load_pdf(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    return list(loader.lazy_load())

def chuck_documents(docs,chunk_size=600, chunk_overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "•", " "]
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )
    return FAISS.from_documents(chunks, embeddings,distance_strategy=DistanceStrategy.COSINE)

def search(vectorstore, query:str, k:int=2):
    return vectorstore.similarity_search_with_score(query, k=k)

def main():
    docs = load_pdf("Deloitte_JD.pdf")
    chunks = chuck_documents(docs)
    vectorstore = build_vectorstore(chunks)
    vectorstore.save_local("faiss_index")
  
    query = "what are the qualifications required for this role?"
    context = "\n\n".join(doc.page_content for doc, score in search(vectorstore, query))

    chat_model = get_chat_model()
    prompt_template = build_prompt()
    interview_questions = generate_interview_questions(chat_model, prompt_template, context, num_categories=5, questions_per_category=5)
    print(interview_questions)
        

if __name__ == "__main__":
    main()