import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20
)

def load_pdf_get_docs(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    docs_lazy = loader.lazy_load()

    docs_list = []
    for doc in docs_lazy:
        docs_list.append(doc)
    return docs_list

docs = load_pdf_get_docs("interview_prep_guide.pdf")
print(f"Number of docs: {len(docs)} in pdf")
chunks = splitter.split_documents(docs)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk: {chunks[0].page_content}")