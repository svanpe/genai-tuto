
import ollama

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader

pdf_path = "kids/en/"
loader = DirectoryLoader(pdf_path, glob="./*.pdf", loader_cls=PyMuPDFLoader)
documents = loader.load()
# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)
print(len(all_splits))

persist_directory="./chromadb/en"
embeddings = OllamaEmbeddings(model="llama3")
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_directory)
#client = chromadb.PersistentClient(path="./chromadb/en")
#collection = client.get_or_create_collection(name="docs")

# persiste the db to disk
vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)



