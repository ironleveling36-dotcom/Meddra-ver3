"""
MedDRA Coding Assistant — FastAPI application.

Endpoints:
  GET  /                  → web search UI (static page)
  GET  /health            → health check
  POST /api/code          → {"text": "...", "top_k": 8} → ranked MedDRA matches
  GET  /api/code?q=...     → same, convenient for quick tests / GET clients
  POST /telegram/webhook   → Telegram updates (webhook mode)
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import engine as engine_mod
from app.config import settings
from app.engine import get_engine
from app.ai_service import AIService, ai_search
from app.telegram_bot import TelegramBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "..", "static")

# Set once the index + model finish loading in the background.
_engine_ready = False


async def _background_load():
    """Load the index and warm up the model WITHOUT blocking server startup, so the
    port binds immediately and Railway's /health probe succeeds right away."""
    global _engine_ready
    try:
        eng = await asyncio.to_thread(get_engine)      # load terms + vectors (~1-2s)
        await asyncio.to_thread(eng.warmup)            # load ONNX model (can be slow)
        _engine_ready = True
        logger.info("Engine loaded and warmed up — ready.")
    except Exception as e:
        logger.error(f"Engine failed to load: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MedDRA Coding Assistant...")
    # Fire-and-forget: don't await, so uvicorn starts serving immediately.
    asyncio.create_task(_background_load())
    try:
        await TelegramBot.start()
    except Exception as e:
        logger.error(f"Telegram start failed: {e}")
    yield
    try:
        await TelegramBot.stop()
    except Exception as e:
        logger.error(f"Telegram stop failed: {e}")
    logger.info("Shutting down.")


app = FastAPI(
    title="MedDRA Coding Assistant",
    description="Natural-language → MedDRA LLT/PT coding via hybrid fuzzy + semantic search.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ──────────────────────────────────────────────────────────
class CodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Free-text term or symptom")
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=settings.MAX_TOP_K)
    ai: bool | None = Field(default=None, description="Use AI accuracy layer (defaults to server setting)")


class Match(BaseModel):
    term: str
    pt: str
    soc: str | None = None
    term_id: int
    level: str
    confidence: float
    semantic_score: float
    lexical_score: float
    match_type: str
    ai_pick: bool = False


class AIInfo(BaseModel):
    used: bool = False
    interpretation: str | None = None
    reason: str | None = None
    expanded_terms: list[str] = []


class CodeResponse(BaseModel):
    query: str
    count: int
    results: list[Match]
    ai: AIInfo = AIInfo()


# ── Routes ──────────────────────────────────────────────────────────
async def _search(text: str, top_k: int, ai: bool | None) -> CodeResponse:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty query")
    top_k = max(1, min(top_k, settings.MAX_TOP_K))
    try:
        eng = get_engine()
    except Exception as e:
        logger.error(f"Engine unavailable: {e}")
        raise HTTPException(status_code=503, detail="Index is still loading or unavailable — please retry in a moment.")

    use_ai = settings.AI_ENABLED and (settings.AI_DEFAULT if ai is None else ai)
    if use_ai:
        try:
            data = await ai_search(eng, text, top_k)
            return CodeResponse(query=data["query"], count=data["count"],
                                results=data["results"], ai=AIInfo(**data["ai"]))
        except Exception as e:
            # AI layer must never block coding results — fall through to plain
            # hybrid search below on ANY failure (network, parsing, provider bug).
            logger.error(f"AI layer failed unexpectedly, falling back to hybrid search: {e}")

    results = eng.search(text, top_k=top_k)
    return CodeResponse(query=text, count=len(results), results=results)


@app.post("/api/code", response_model=CodeResponse)
async def code_post(req: CodeRequest):
    return await _search(req.text, req.top_k, req.ai)


@app.get("/api/code", response_model=CodeResponse)
async def code_get(q: str, top_k: int = settings.DEFAULT_TOP_K, ai: bool | None = None):
    return await _search(q, top_k, ai)


@app.get("/health")
def health():
    # Must NOT trigger the (slow) engine load — always answer fast with 200 so the
    # platform healthcheck passes while the index warms up in the background.
    eng = engine_mod._engine
    return {
        "status": "healthy",
        "ready": _engine_ready,
        "terms_loaded": len(eng.terms) if eng else 0,
        "model_loaded": bool(eng and eng._model is not None),
        "telegram": "enabled" if settings.TELEGRAM_ENABLED else "disabled",
        "ai_layer": "enabled" if settings.AI_ENABLED else "disabled",
    }


@app.get("/api/ai-health")
async def ai_health():
    """Lightweight, cached check of whether the AI accuracy layer is reachable.
    Powers the green/red AI status signal and the AI Suggest side tab on the web UI."""
    if not settings.AI_ENABLED:
        return {"configured": False, "live": False, "detail": "AI disabled — no AI_API_KEY set"}
    return await AIService.check_health()


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if settings.TELEGRAM_WEBHOOK_SECRET:
        if request.headers.get("X-Telegram-Bot-Api-Secret-Token", "") != settings.TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")
    try:
        await TelegramBot.handle_update(await request.json())
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    return {"ok": True}


@app.get("/")
def index():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"service": "MedDRA Coding Assistant", "docs": "/docs", "api": "/api/code?q=bleeding"})


# Serve any other static assets (favicon, etc.) if present.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
