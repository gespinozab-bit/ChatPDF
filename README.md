# Week 8 - Chat with your PDF

Aplicacion web minima tipo "Chat with your PDF" construida con FastAPI, LangChain, OpenAI y PGVector.

La app permite subir un PDF, dividirlo en chunks, generar embeddings, guardarlos en PostgreSQL con pgvector, recuperar los chunks mas relevantes y generar una respuesta con un LLM usando solo el contexto recuperado. Si OpenAI no tiene cuota disponible, la app cae a un modo local extractivo para seguir funcionando sin bloquear la demo.

## Arquitectura

```text
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> OpenAIEmbeddings
    -> PGVector -> similarity_search -> ChatOpenAI -> respuesta + chunks
```

Modo de respaldo:

```text
PDF -> PyPDFLoader -> RecursiveCharacterTextSplitter -> busqueda lexica local
    -> respuesta extractiva usando solo chunks recuperados
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

Para incluir las herramientas de desarrollo, incluida la herramienta SAST:

```bash
uv sync --dev
```

## SAST con Bandit

Este proyecto integra [Bandit](https://bandit.readthedocs.io/) como herramienta SAST del ecosistema DevSecOps para detectar patrones inseguros en codigo Python.

Ejecuta el analisis localmente con:

```bash
uv run bandit -c pyproject.toml -r app ingestor.py query.py
```

El mismo analisis esta definido en `.github/workflows/sast.yml` para ejecutarse en `push` a `main` y en cada `pull_request`.

## SAST con SonarQube

Tambien se incluye SonarQube como herramienta SAST con dashboard web. Esta opcion es util para explicar DevSecOps en clase porque muestra metricas visuales de seguridad, mantenibilidad, bugs, code smells y duplicacion.

Archivos usados:

- `docker-compose.yml`: levanta SonarQube y un contenedor `sonar-scanner`.
- `sonar-project.properties`: define que archivos del proyecto se analizan.
- `.env`: guarda el token local de SonarQube en `SONAR_TOKEN`.

Levanta SonarQube:

```bash
docker compose up -d sonarqube
```

Abre el dashboard:

```text
http://localhost:9000
```

Credenciales iniciales:

```text
usuario: admin
password: admin
```

SonarQube pedira cambiar la contrasena la primera vez. Luego genera un token en:

```text
My Account -> Security -> Generate Tokens
```

Agrega el token al archivo `.env`:

```bash
SONAR_TOKEN=replace-with-your-sonarqube-token
```

Ejecuta el analisis SAST con el scanner:

```bash
docker compose --profile manual run --rm sonar-scanner
```

Al terminar, los resultados aparecen en:

```text
http://localhost:9000/dashboard?id=week-8-rag
```

Para detener SonarQube:

```bash
docker compose down
```

### Explicacion para clase

SAST significa Static Application Security Testing. Es una practica DevSecOps que analiza el codigo fuente sin ejecutar la aplicacion. En este proyecto, SonarQube revisa archivos Python como `app/main.py`, `app/rag.py`, `ingestor.py` y `query.py`.

El flujo es:

```text
Codigo fuente -> sonar-scanner -> SonarQube -> Dashboard de resultados
```

La idea es encontrar riesgos antes de desplegar. Por ejemplo, si un desarrollador agrega codigo inseguro, SonarQube puede reportarlo como vulnerabilidad, bug o code smell. Asi el equipo corrige el problema antes de que llegue a produccion.

En este proyecto quedan dos niveles de SAST:

- `Bandit`: rapido, especializado en Python y facil de ejecutar en terminal o GitHub Actions.
- `SonarQube`: visual, con dashboard local para explicar el analisis y presentar resultados.

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
SONAR_TOKEN=replace-with-your-sonarqube-token
```

Valores importantes:

- `DATABASE_URL`: conexion usada por PGVector.
- `PGVECTOR_COLLECTION`: nombre unico de coleccion para la app y los scripts.
- `OPENAI_EMBEDDING_MODEL`: modelo para embeddings.
- `OPENAI_CHAT_MODEL`: modelo para generar la respuesta final.
- `CHUNK_SIZE` y `CHUNK_OVERLAP`: controlan la division del PDF.
- `TOP_K`: numero por defecto de chunks recuperados.
- `MAX_UPLOAD_MB`: limite de subida desde la UI.
- `SONAR_TOKEN`: token local usado por `sonar-scanner` para enviar resultados a SonarQube.

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

Si OpenAI no tiene cuota o PGVector no esta disponible, el PDF se mantiene en memoria como `local_memory_fallback`. En ese modo puedes preguntar durante la sesion actual del servidor, pero debes volver a subir el PDF si reinicias Uvicorn.

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

## Problemas comunes

Si ves un error de `OPENAI_API_KEY`, revisa que exista un archivo `.env` real en la raiz del proyecto. `.env.example` es solo una plantilla y no debe contener secretos.

Si OpenAI responde `insufficient_quota`, la API key fue leida correctamente, pero la cuenta no tiene cuota, creditos o billing disponible para embeddings/LLM. La app usa el modo local extractivo como respaldo para que puedas seguir probando, aunque la calidad de respuesta sera menor que con el LLM.

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
