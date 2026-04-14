const uploadForm = document.querySelector("#upload-form");
const askForm = document.querySelector("#ask-form");
const uploadStatus = document.querySelector("#upload-status");
const modeStatus = document.querySelector("#mode-status");
const answerBox = document.querySelector("#answer");
const chunksBox = document.querySelector("#chunks");

function setStatus(message, type = "") {
  uploadStatus.textContent = message;
  uploadStatus.className = `status ${type}`.trim();
}

function setMode(mode) {
  modeStatus.textContent = `Modo: ${mode}`;
}

function setBusy(form, busy) {
  for (const element of form.elements) {
    element.disabled = busy;
  }
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || "Ocurrio un error inesperado.";
  } catch {
    return "Ocurrio un error inesperado.";
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = document.querySelector("#pdf-file");
  if (!fileInput.files.length) {
    setStatus("Selecciona un PDF.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  setBusy(uploadForm, true);
  setStatus("Procesando PDF...");
  answerBox.textContent = "La respuesta aparecera aqui.";
  chunksBox.textContent = "Los fragmentos relevantes apareceran aqui.";

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }
    const data = await response.json();
    setMode(data.mode);
    setStatus(
      `PDF procesado: ${data.filename}. Paginas: ${data.pages}. Chunks: ${data.chunks}. Coleccion: ${data.collection_name}.`,
      "success"
    );
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(uploadForm, false);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = document.querySelector("#question").value.trim();
  const topK = Number(document.querySelector("#top-k").value || 4);
  if (!question) {
    answerBox.textContent = "Escribe una pregunta.";
    return;
  }

  setBusy(askForm, true);
  answerBox.textContent = "Buscando contexto y generando respuesta...";
  chunksBox.textContent = "";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK }),
    });
    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const data = await response.json();
    setMode(data.mode);
    answerBox.textContent = data.answer;
    renderChunks(data.chunks);
  } catch (error) {
    answerBox.textContent = error.message;
    chunksBox.textContent = "No hay chunks para mostrar.";
  } finally {
    setBusy(askForm, false);
  }
});

function renderChunks(chunks) {
  chunksBox.innerHTML = "";
  if (!chunks.length) {
    chunksBox.textContent = "No se recuperaron chunks.";
    return;
  }

  for (const chunk of chunks) {
    const article = document.createElement("article");
    article.className = "chunk";

    const meta = document.createElement("div");
    meta.className = "chunk-meta";
    const page = chunk.metadata.page_number ?? chunk.metadata.page ?? "desconocida";
    const source = chunk.metadata.source_file || chunk.metadata.source || "PDF";
    meta.textContent = `Chunk ${chunk.index} | pagina: ${page} | fuente: ${source}`;

    const content = document.createElement("p");
    content.className = "chunk-content";
    content.textContent = chunk.content;

    article.append(meta, content);
    chunksBox.append(article);
  }
}
