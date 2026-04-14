import os
import tempfile
from dataclasses import dataclass
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AuthenticationError, RateLimitError

from .config import Settings, settings


class RAGError(Exception):
    """Base error for user-facing RAG failures."""


class InvalidPDFError(RAGError):
    pass


class EmptyPDFError(RAGError):
    pass


class NoDocumentError(RAGError):
    pass


class ExternalServiceError(RAGError):
    pass


@dataclass
class ProcessedDocument:
    filename: str
    pages: int
    chunks: int
    collection_name: str


class RAGService:
    def __init__(self, app_settings: Settings = settings):
        self.settings = app_settings
        self._has_document = False
        self._current_filename: str | None = None

    def process_pdf(self, file_bytes: bytes, filename: str) -> ProcessedDocument:
        if not file_bytes:
            raise InvalidPDFError("El archivo esta vacio.")

        temp_path = self._write_temp_pdf(file_bytes)
        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
        except Exception as exc:
            raise InvalidPDFError("No se pudo leer el PDF. Verifica que el archivo sea valido.") from exc
        finally:
            self._remove_temp_file(temp_path)

        if not pages or not any(page.page_content.strip() for page in pages):
            raise EmptyPDFError("El PDF no contiene texto extraible.")

        for page in pages:
            page.metadata["source_file"] = filename
            page.metadata["source"] = filename
            if isinstance(page.metadata.get("page"), int):
                page.metadata["page_number"] = page.metadata["page"] + 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise EmptyPDFError("El PDF no genero chunks con texto util.")

        try:
            vector_store = self._reset_vector_store()
            vector_store.add_documents(documents=chunks)
        except ExternalServiceError:
            raise
        except AuthenticationError as exc:
            raise ExternalServiceError(
                "OpenAI rechazo la API key. Verifica que OPENAI_API_KEY sea valida y este activa."
            ) from exc
        except RateLimitError as exc:
            raise ExternalServiceError(
                "OpenAI rechazo la solicitud por cuota insuficiente. Revisa billing, creditos o limites de tu cuenta."
            ) from exc
        except Exception as exc:
            raise ExternalServiceError(
                "Fallo al generar embeddings o guardar en PGVector. Revisa OpenAI y PostgreSQL."
            ) from exc

        self._has_document = True
        self._current_filename = filename

        return ProcessedDocument(
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            collection_name=self.settings.collection_name,
        )

    def answer_question(self, question: str, top_k: int) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("La pregunta no puede estar vacia.")
        if not self._has_document:
            raise NoDocumentError("Primero sube y procesa un PDF.")

        try:
            vector_store = self._vector_store()
            docs = vector_store.similarity_search(query=clean_question, k=top_k)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError("Fallo la busqueda en el vector store.") from exc

        docs = [doc for doc in docs if doc.page_content.strip()]
        if not docs:
            return {
                "answer": "No se encontro suficiente informacion en el PDF para responder.",
                "chunks": [],
            }

        context = self._format_context(docs)
        try:
            self._ensure_openai_api_key()
            llm = ChatOpenAI(model=self.settings.chat_model, temperature=0)
            response = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Eres un asistente RAG. Responde en espanol usando solo el contexto recuperado. "
                            "Si el contexto no contiene informacion suficiente para responder, di exactamente: "
                            "'No se encontro suficiente informacion en el PDF para responder.' "
                            "No inventes datos ni uses conocimiento externo."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Pregunta: {clean_question}\n\n"
                            f"Contexto recuperado:\n{context}\n\n"
                            "Respuesta:"
                        )
                    ),
                ]
            )
        except AuthenticationError as exc:
            raise ExternalServiceError(
                "OpenAI rechazo la API key. Verifica que OPENAI_API_KEY sea valida y este activa."
            ) from exc
        except RateLimitError as exc:
            raise ExternalServiceError(
                "OpenAI rechazo la solicitud por cuota insuficiente. Revisa billing, creditos o limites de tu cuenta."
            ) from exc
        except Exception as exc:
            raise ExternalServiceError("Fallo la llamada al modelo de lenguaje.") from exc

        return {
            "answer": str(response.content).strip(),
            "chunks": [
                {
                    "index": index,
                    "content": doc.page_content,
                    "metadata": self._clean_metadata(doc.metadata),
                }
                for index, doc in enumerate(docs, start=1)
            ],
        }

    def _embeddings(self) -> OpenAIEmbeddings:
        self._ensure_openai_api_key()
        return OpenAIEmbeddings(model=self.settings.embedding_model)

    @staticmethod
    def _ensure_openai_api_key() -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or api_key == "replace-with-your-openai-api-key":
            raise ExternalServiceError(
                "Falta OPENAI_API_KEY. Reemplaza el valor de OPENAI_API_KEY en tu archivo .env local."
            )

    def _vector_store(self) -> PGVector:
        return PGVector(
            embeddings=self._embeddings(),
            collection_name=self.settings.collection_name,
            connection=self.settings.database_url,
            use_jsonb=True,
        )

    def _reset_vector_store(self) -> PGVector:
        vector_store = self._vector_store()
        try:
            vector_store.delete_collection()
        except Exception:
            # The collection may not exist on the first upload.
            pass
        return self._vector_store()

    @staticmethod
    def _write_temp_pdf(file_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            return temp_file.name

    @staticmethod
    def _remove_temp_file(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    @staticmethod
    def _format_context(docs: list[Any]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            metadata = RAGService._clean_metadata(doc.metadata)
            page = metadata.get("page_number") or metadata.get("page", "desconocida")
            source = metadata.get("source_file") or metadata.get("source") or "PDF"
            parts.append(
                f"[Chunk {index} | pagina={page} | fuente={source}]\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean
