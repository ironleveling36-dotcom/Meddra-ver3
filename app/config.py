"""Configuration for the MedDRA Coding Assistant (all env-driven)."""
import os

from dotenv import load_dotenv

load_dotenv()


def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # CORS (comma-separated origins; * allows all)
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    # Default number of results to return
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "8"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "40"))

    # Semantic (AI embedding) search. Set to "false" for LITE MODE — fuzzy/lexical
    # matching only, no ONNX model, no ~400MB RAM, no internet needed. Ideal for
    # shared cPanel hosting. Defaults on for full-quality deployments.
    ENABLE_SEMANTIC: bool = _as_bool(os.getenv("ENABLE_SEMANTIC"), True)

    # Public URL (auto-detected on Railway) — used to register the Telegram webhook
    _railway_domain: str = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
    PUBLIC_URL: str = (
        os.getenv("PUBLIC_URL")
        or (f"https://{_railway_domain}" if _railway_domain else "")
    ).rstrip("/")

    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_MODE: str = os.getenv("TELEGRAM_MODE", "polling").strip().lower()
    TELEGRAM_WEBHOOK_SECRET: str = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

    @property
    def TELEGRAM_ENABLED(self) -> bool:
        return bool(self.TELEGRAM_BOT_TOKEN)

    # AI accuracy layer (optional) — any OpenAI-compatible /chat/completions API.
    # Interprets tricky lay phrases/idioms and re-ranks the real MedDRA candidates;
    # it NEVER invents codes, it only picks among the hybrid search results.
    AI_API_BASE_URL: str = os.getenv("AI_API_BASE_URL", "https://api.hcnsec.cn/v1").rstrip("/")
    AI_API_KEY: str = os.getenv("AI_API_KEY", "").strip()
    # "auto" lets the provider route to its best available model; set AI_MODEL to
    # a specific one (e.g. "kimi-k2") if you want to pin it.
    AI_MODEL: str = os.getenv("AI_MODEL", "auto").strip()
    AI_TIMEOUT: float = float(os.getenv("AI_TIMEOUT", "20"))
    # When true (and a key is set), the API/UI use AI by default.
    AI_DEFAULT: bool = _as_bool(os.getenv("AI_DEFAULT"), True)

    @property
    def AI_ENABLED(self) -> bool:
        return bool(self.AI_API_KEY)


settings = Settings()
