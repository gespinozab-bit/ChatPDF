import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from math import sqrt
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
        self._local_chunks: list[Any] = []
        self._storage_mode = "pgvector"

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
            self._use_local_fallback(chunks, filename)
        except AuthenticationError:
            self._use_local_fallback(chunks, filename)
        except RateLimitError:
            self._use_local_fallback(chunks, filename)
        except Exception:
            self._use_local_fallback(chunks, filename)
        else:
            self._local_chunks = chunks
            self._storage_mode = "pgvector"

        self._has_document = True
        self._current_filename = filename

        return ProcessedDocument(
            filename=filename,
            pages=len(pages),
            chunks=len(chunks),
            collection_name=self._collection_label(),
        )

    def answer_question(self, question: str, top_k: int) -> dict[str, Any]:
        clean_question = question.strip()
        if not clean_question:
            raise RAGError("La pregunta no puede estar vacia.")
        if not self._has_document:
            raise NoDocumentError("Primero sube y procesa un PDF.")

        if self._storage_mode == "local":
            docs = self._local_similarity_search(clean_question, top_k)
        else:
            try:
                vector_store = self._vector_store()
                docs = vector_store.similarity_search(query=clean_question, k=top_k)
            except Exception:
                docs = self._local_similarity_search(clean_question, top_k)

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
            answer = self._extractive_answer(clean_question, docs)
        except RateLimitError:
            answer = self._extractive_answer(clean_question, docs)
        except Exception:
            answer = self._extractive_answer(clean_question, docs)
        else:
            answer = str(response.content).strip()

        return {
            "answer": answer,
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

    def _use_local_fallback(self, chunks: list[Any], filename: str) -> None:
        self._local_chunks = chunks
        self._current_filename = filename
        self._storage_mode = "local"

    def _collection_label(self) -> str:
        if self._storage_mode == "local":
            return "local_memory_fallback"
        return self.settings.collection_name

    def _local_similarity_search(self, query: str, top_k: int) -> list[Any]:
        query_vector = self._term_vector(query)
        if not query_vector:
            return self._local_chunks[:top_k]

        scored = []
        required_coverage = self._required_coverage(query_vector)
        for chunk in self._local_chunks:
            chunk_vector = self._term_vector(chunk.page_content)
            score = self._cosine_similarity(query_vector, chunk_vector)
            coverage = self._term_coverage(query_vector, chunk_vector)
            if score >= 0.035 and coverage >= required_coverage:
                scored.append((score + coverage, chunk))

        if not scored:
            return []

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _extractive_answer(self, question: str, docs: list[Any]) -> str:
        sentences = []
        question_vector = self._term_vector(question)
        required_coverage = self._required_coverage(question_vector)
        for doc in docs:
            for sentence in re.split(r"(?<=[.!?])\s+", doc.page_content.strip()):
                clean_sentence = sentence.strip()
                if len(clean_sentence) < 30:
                    continue
                if not question_vector:
                    sentences.append((1.0, clean_sentence))
                    continue
                sentence_vector = self._term_vector(clean_sentence)
                score = self._cosine_similarity(question_vector, sentence_vector)
                coverage = self._term_coverage(question_vector, sentence_vector)
                if score >= 0.08 and coverage >= required_coverage:
                    sentences.append((score + coverage, clean_sentence))

        if not sentences:
            return "No se encontro suficiente informacion en el PDF para responder."

        sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in sentences[:3]]
        return (
            "Modo local sin LLM: con base en los chunks recuperados, la informacion mas relevante es: "
            + " ".join(selected)
        )

    @staticmethod
    def _term_vector(text: str) -> Counter[str]:
        stop_words = {
            "a",
            "al",
            "cual",
            "de",
            "del",
            "documento",
            "el",
            "en",
            "es",
            "la",
            "las",
            "lo",
            "los",
            "por",
            "que",
            "se",
            "sobre",
            "trata",
            "un",
            "una",
            "y",
        }
        tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]{3,}", text.lower())
        return Counter(token for token in tokens if token not in stop_words)

    @staticmethod
    def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        common = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = sqrt(sum(value * value for value in left.values()))
        right_norm = sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _term_coverage(query_vector: Counter[str], candidate_vector: Counter[str]) -> float:
        if not query_vector or not candidate_vector:
            return 0.0
        matches = set(query_vector) & set(candidate_vector)
        return len(matches) / len(set(query_vector))

    @staticmethod
    def _required_coverage(query_vector: Counter[str]) -> float:
        if len(query_vector) <= 2:
            return 1.0
        if len(query_vector) <= 4:
            return 0.75
        return 0.6

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
