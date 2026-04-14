# Week 8 - Chat with your PDF

Aplicacion web minima tipo "Chat with your PDF" construida con FastAPI, LangChain, OpenAI y PGVector.

La app permite subir un PDF, dividirlo en chunks, generar embeddings, guardarlos en PostgreSQL con pgvector, recuperar los chunks mas relevantes y generar una respuesta con un LLM usando solo el contexto recuperado.

## Arquitectura

```text
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> OpenAIEmbeddings
    -> PGVector -> similarity_search -> ChatOpenAI -> respuesta + chunks
```

Estructura principal:

```text
app/
  main.py              # Rutas FastAPI, UI, upload y preguntas
  rag.py               # Carga PDF, chunking, embeddings, PGVector y LLM
  schemas.py           # Modelos Pydantic para request/response
  config.py            # Variables de entorno y valores por defecto
  templates/index.html # UI
  static/style.css     # Estilos
  static/app.js        # Logica del navegador
```

Los scripts `ingestor.py` y `query.py` siguen disponibles como ejemplos de consola, pero ahora reutilizan la misma configuracion que la app web.

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker, o una instancia PostgreSQL con la extension `pgvector`
- Una API key de OpenAI

## Instalacion

Instala las dependencias:

```bash
uv sync
```

## Variables de entorno

Crea un archivo `.env` en la raiz del proyecto:

```bash
OPENAI_API_KEY=replace-with-your-openai-api-key
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/week_8
PGVECTOR_COLLECTION=pdf_chat_documents
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
MAX_UPLOAD_MB=20
```

Valores importantes:

- `DATABASE_URL`: conexion usada por PGVector.
- `PGVECTOR_COLLECTION`: nombre unico de coleccion para la app y los scripts.
- `OPENAI_EMBEDDING_MODEL`: modelo para embeddings.
- `OPENAI_CHAT_MODEL`: modelo para generar la respuesta final.
- `CHUNK_SIZE` y `CHUNK_OVERLAP`: controlan la division del PDF.
- `TOP_K`: numero por defecto de chunks recuperados.
- `MAX_UPLOAD_MB`: limite de subida desde la UI.

## PostgreSQL con pgvector

Puedes levantar una base local con Docker:

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=week_8 \
  -p 5433:5432 \
  pgvector/pgvector:pg18-trixie
```

La configuracion por defecto conecta a:

```text
postgresql+psycopg://postgres:postgres@localhost:5433/week_8
```

## Ejecutar la app web

```bash
uv run uvicorn app.main:app --reload
```

Abre:

```text
http://127.0.0.1:8000
```

## Subir un PDF

1. En la UI, selecciona un archivo `.pdf`.
2. Haz clic en `Procesar PDF`.
3. La app carga el PDF con `PyPDFLoader`.
4. Si el PDF no tiene texto extraible, devuelve un error.
5. Si se procesa bien, muestra cantidad de paginas, chunks y coleccion PGVector.

Al subir un PDF nuevo, la app limpia la coleccion configurada en `PGVECTOR_COLLECTION` y guarda los chunks del nuevo documento. Esto evita mezclar respuestas entre PDFs.

## Hacer preguntas

1. Escribe una pregunta sobre el PDF procesado.
2. Ajusta `Chunks` si quieres recuperar mas o menos contexto.
3. Haz clic en `Preguntar`.
4. La app hace `similarity_search`, pasa los chunks recuperados al LLM y muestra:
   - la respuesta final,
   - los chunks recuperados,
   - metadata disponible como pagina, fuente y `start_index`.

El prompt del LLM obliga a responder solo con el contexto recuperado. Si el contexto no contiene informacion suficiente, debe responder:

```text
No se encontro suficiente informacion en el PDF para responder.
```

## Manejo de errores

La API maneja casos basicos:

- archivo invalido o no PDF,
- PDF sin texto extraible,
- pregunta antes de procesar un documento,
- fallos al generar embeddings,
- fallos al guardar o consultar PGVector,
- fallos al llamar al LLM.

## Endpoints

```text
GET  /
POST /api/upload
POST /api/ask
```

Ejemplo con `curl` para subir un PDF:

```bash
curl -F "file=@docs/codigo-de-trabajo.pdf" http://127.0.0.1:8000/api/upload
```

Ejemplo para preguntar:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"De que trata el documento?","top_k":4}'
```

## Scripts de consola

Ingestar el PDF de ejemplo:

```bash
uv run ingestor.py
```

Consultar por consola:

```bash
uv run query.py
```
