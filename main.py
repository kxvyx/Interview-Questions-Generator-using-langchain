import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS, DistanceStrategy

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

    query = "what are the qualifications required for this role?"
    for doc, score in search(vectorstore, query):
        print(f"Score: {score}, Content: {doc.page_content}")

if __name__ == "__main__":
    main()