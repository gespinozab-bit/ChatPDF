from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .rag import (
    EmptyPDFError,
    ExternalServiceError,
    InvalidPDFError,
    NoDocumentError,
    RAGError,
    RAGService,
)
from .schemas import AnswerResponse, QuestionRequest, UploadResponse


app = FastAPI(title="Chat with your PDF", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")
rag_service = RAGService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sube un archivo PDF valido.")
    if file.content_type and file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    file_bytes = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"El PDF supera el limite de {settings.max_upload_mb} MB.",
        )

    try:
        result = rag_service.process_pdf(file_bytes=file_bytes, filename=file.filename)
        return UploadResponse(**result.__dict__)
    except (InvalidPDFError, EmptyPDFError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest):
    try:
        result = rag_service.answer_question(payload.question, payload.top_k)
        return AnswerResponse(**result)
    except NoDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ExternalServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RAGError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
