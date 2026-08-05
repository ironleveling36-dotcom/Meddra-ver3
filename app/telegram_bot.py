"""
Telegram bot for the MedDRA Coding Assistant.

Send any medical term / symptom / product complaint and the bot replies with the
top matching MedDRA LLT/PT terms and confidence scores. Reuses the same engine as
the web app and REST API. Only httpx required — no bot framework.

Modes (TELEGRAM_MODE): "polling" (default, zero-config) or "webhook".
"""
import asyncio
import logging
from typing import Optional

import httpx

from app.config import settings
from app.engine import get_engine

logger = logging.getLogger(__name__)

HELP = (
    "🩺 *MedDRA Coding Assistant*\n\n"
    "Send me any medical term, symptom, product complaint, abbreviation, or even a "
    "misspelling — I'll find the closest MedDRA *LLT* and *PT* terms with confidence "
    "scores.\n\n"
    "Examples:\n"
    "• `bleeding`\n"
    "• `drug not working`\n"
    "• `tablet is hard`\n"
    "• `SOB`\n\n"
    "Commands: /start, /help"
)


def format_results(query: str, results: list[dict], ai: dict | None = None) -> str:
    header = ""
    if ai and ai.get("used") and ai.get("interpretation"):
        header = f"🤖 _AI: {ai['interpretation']}_\n\n"
    if not results:
        return header + f"No MedDRA match found for *{query}*. Try rephrasing or add more detail."
    lines = [header + f"🔎 Top MedDRA matches for *{query}*:\n"]
    for i, r in enumerate(results, 1):
        pick = " 🤖" if r.get("ai_pick") else ""
        # Lead with the LLT (the granular term being coded); PT shown underneath.
        pt_line = ""
        if r.get("pt") and r["pt"].lower() != r["term"].lower():
            pt_line = f"\n   ↳ PT: {r['pt']}"
        soc = f"\n   _SOC: {r['soc']}_" if r.get("soc") else ""
        lines.append(
            f"{i}. *{r['term']}*  [{r['level']}]  ({r['confidence']}%){pick}"
            f"{pt_line}\n   ID: {r['term_id']}{soc}"
        )
    return "\n".join(lines)


class TelegramBot:
    _polling_task: Optional[asyncio.Task] = None
    _stop = False

    @classmethod
    def _url(cls, method: str) -> str:
        return f"https://api.telegram.org/bot{settings.TELEGRAM_BOT_TOKEN}/{method}"

    @classmethod
    async def _call(cls, method: str, payload: dict, timeout: float = 30.0) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(cls._url(method), json=payload)
                data = r.json()
                if not data.get("ok"):
                    logger.warning(f"Telegram {method} not ok: {data}")
                return data
        except Exception as e:
            logger.error(f"Telegram {method} failed: {e}")
            return None

    @classmethod
    async def send(cls, chat_id: int, text: str) -> None:
        for start in range(0, len(text), 4000):
            await cls._call("sendMessage", {
                "chat_id": chat_id,
                "text": text[start:start + 4000],
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            })

    @classmethod
    async def handle_update(cls, update: dict) -> None:
        msg = update.get("message") or update.get("edited_message")
        if not msg:
            return
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()
        if chat_id is None or not text:
            return

        if text.lower() in ("/start", "/help", "start", "help"):
            await cls.send(chat_id, HELP)
            return

        await cls._call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10.0)
        try:
            if settings.AI_ENABLED and settings.AI_DEFAULT:
                from app.ai_service import ai_search
                data = await ai_search(get_engine(), text, settings.DEFAULT_TOP_K)
                await cls.send(chat_id, format_results(text, data["results"], data.get("ai")))
            else:
                # engine.search is CPU-bound → run in a thread so we don't block the loop.
                results = await asyncio.to_thread(get_engine().search, text, settings.DEFAULT_TOP_K)
                await cls.send(chat_id, format_results(text, results))
        except Exception as e:
            logger.error(f"Search error: {e}")
            await cls.send(chat_id, "⚠️ Something went wrong. Please try again.")

    # ── Polling ─────────────────────────────────────────────────────
    @classmethod
    async def _poll_loop(cls) -> None:
        logger.info("Telegram polling started.")
        offset = 0
        # Skip backlog on startup.
        try:
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(cls._url("getUpdates"), json={"offset": -1, "timeout": 0})
                res = r.json().get("result", [])
                if res:
                    offset = res[-1]["update_id"] + 1
        except Exception:
            pass

        while not cls._stop:
            try:
                async with httpx.AsyncClient(timeout=40.0) as c:
                    r = await c.post(cls._url("getUpdates"), json={
                        "offset": offset, "timeout": 30,
                        "allowed_updates": ["message", "edited_message"],
                    })
                    data = r.json()
                if not data.get("ok"):
                    await asyncio.sleep(3)
                    continue
                for upd in data.get("result", []):
                    offset = max(offset, upd["update_id"] + 1)
                    try:
                        await cls.handle_update(upd)
                    except Exception as e:
                        logger.error(f"handle_update error: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)
        logger.info("Telegram polling stopped.")

    # ── Lifecycle ───────────────────────────────────────────────────
    @classmethod
    async def start(cls) -> None:
        if not settings.TELEGRAM_ENABLED:
            logger.info("Telegram disabled (no TELEGRAM_BOT_TOKEN).")
            return
        me = await cls._call("getMe", {}, timeout=10.0)
        if not me or not me.get("ok"):
            logger.error("Invalid TELEGRAM_BOT_TOKEN; bot not started.")
            return
        logger.info(f"Telegram bot @{me['result'].get('username')} ready.")
        await cls._call("setMyCommands", {"commands": [
            {"command": "start", "description": "How to use the coder"},
            {"command": "help", "description": "Show help"},
        ]})

        if settings.TELEGRAM_MODE == "webhook" and settings.PUBLIC_URL:
            url = f"{settings.PUBLIC_URL}/telegram/webhook"
            payload = {"url": url, "allowed_updates": ["message", "edited_message"]}
            if settings.TELEGRAM_WEBHOOK_SECRET:
                payload["secret_token"] = settings.TELEGRAM_WEBHOOK_SECRET
            data = await cls._call("setWebhook", payload)
            if data and data.get("ok"):
                logger.info(f"Webhook set: {url}")
                return
            logger.error("setWebhook failed; falling back to polling.")

        await cls._call("deleteWebhook", {"drop_pending_updates": False})
        cls._stop = False
        cls._polling_task = asyncio.create_task(cls._poll_loop())

    @classmethod
    async def stop(cls) -> None:
        cls._stop = True
        if cls._polling_task:
            cls._polling_task.cancel()
            try:
                await cls._polling_task
            except (asyncio.CancelledError, Exception):
                pass
            cls._polling_task = None
