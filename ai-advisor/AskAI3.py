import chromadb
import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

persist_directory="./chromadb/en"
embeddings = OllamaEmbeddings(model="llama3")

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

#client = chromadb.PersistentClient(path="./chromadb/en")
#collection = client.get_collection(name="docs")

retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents("risk class")

# an example prompt
for document in docs:
    print(f"  {document.metadata['source']} , {document.page_content}  ")

