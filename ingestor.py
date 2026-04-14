from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import settings

file_path = "./docs/codigo-de-trabajo.pdf"

loader = PyPDFLoader(file_path)

# carga el PDF
docs = loader.load() # [Document, Document, Document]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Total pages: {len(docs)}")
print(f"Total chunks: {len(all_splits)}")

embeddings = OpenAIEmbeddings(model=settings.embedding_model)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=settings.collection_name,
    connection=settings.database_url,
    use_jsonb=True,
)

# Store documents in the vector store   
ids = vector_store.add_documents(documents=all_splits)
