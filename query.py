from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import settings

embeddings = OpenAIEmbeddings(model=settings.embedding_model)
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=settings.collection_name,
    connection=settings.database_url,
    use_jsonb=True,
)

# query the vector store
query = input("Ingresa tu consulta: ")

results = vector_store.similarity_search(query=query)



print(results)
