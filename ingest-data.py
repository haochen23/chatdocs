"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader, BSHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

import os
import openai


def find_html_files(directory):
    html_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files


def ingest_docs():
    """Get documents from web pages."""
    html_files = find_html_files("/Users/ericchen/test/learn.microsoft.com/")

    raw_documents = []
    for f in html_files:
        loader = BSHTMLLoader(f)
        raw_documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    # print(os.getcwd())


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    print(os.getenv("OPENAI_API_KEY"))
    ingest_docs()
