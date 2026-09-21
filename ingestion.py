from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

if __name__ == "__main__":
    print("Loading documents from text file...")

    loader = TextLoader(
        "C:\\Users\\hanee\\OneDrive\\Documents\\Agentic Ai\\langchain-course\\mediumblog1.txt",
        encoding="utf-8",
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s)")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    print("Creating embeddings and storing in Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        texts, embeddings, persist_directory="./chroma_db"
    )
    print("Finished, vector store saved to ./chroma_db")