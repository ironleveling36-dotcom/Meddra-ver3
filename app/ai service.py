"""
AI accuracy layer for higher-accuracy MedDRA coding.

Talks to any OpenAI-compatible `/chat/completions` endpoint (configured via
AI_API_BASE_URL / AI_API_KEY / AI_MODEL — defaults to https://api.hcnsec.cn/v1
with model "auto", which lets the provider route to its best model, e.g. Kimi K2).

The AI NEVER invents MedDRA codes. It only:
  1. interprets the user's lay phrase into clinical meaning,
  2. re-ranks the *real* candidate terms our hybrid search already found, and
  3. optionally suggests better clinical search terms when none of the candidates
     fit (e.g. an idiom like "nerve dancing"), which we then search ourselves.

Everything is grounded in the actual MedDRA index, so IDs are always valid.
Optional: activates only when AI_API_KEY is set; otherwise callers fall back
to pure hybrid search.
"""
import json
import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

CHAT_URL_TMPL = "{base}/chat/completions"
MODELS_URL_TMPL = "{base}/models"

# Cache the health check briefly so the UI can poll frequently without
# spamming the provider or burning quota.
_health_cache: dict | None = None
_health_cache_at: float = 0.0
HEALTH_CACHE_TTL = 15  # seconds

# Cache AI refine results for 10 minutes (same query = same answer)
_refine_cache: dict[str, dict | None] = {}
_refine_cache_max_size = 256

SYSTEM_PROMPT = """Quickly pick the best MedDRA codes from the list. JSON only:
{"interpretation": string, "ranked_term_ids": [int], "need_more": bool, "suggested_terms": [string], "reason": string}"""


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.AI_API_KEY}",
        "Content-Type": "application/json",
    }


def _classify_error(status_code: int, body: str) -> str:
    """Turn an HTTP status/body into a short, user-facing reason."""
    if status_code in (401, 403):
        return "AI key invalid or dead"
    if status_code == 429:
        return "AI quota exceeded / rate limited"
    if status_code == 404:
        return "AI model or endpoint not found"
    if status_code >= 500:
        return "AI provider is having issues"
    low = (body or "").lower()
    if "quota" in low or "insufficient" in low or "balance" in low:
        return "AI quota exceeded"
    if "invalid" in low and "key" in low:
        return "AI key invalid or dead"
    return f"AI request failed (HTTP {status_code})"


def _extract_text(data: dict) -> str:
    """Pull the assistant's text out of an OpenAI-style response, tolerating the
    shape variations different providers/proxies use (list-of-parts content,
    reasoning_content, etc.)."""
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        # Some providers return content as [{"type":"text","text":"..."}]
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            elif isinstance(part, str):
                parts.append(part)
        joined = "".join(parts).strip()
        if joined:
            return joined
    # Some reasoning models put the real answer here instead of content.
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return ""


class AIService:
    @staticmethod
    def _candidate_block(candidates: list[dict]) -> str:
        lines = []
        for c in candidates:
            lines.append(f'{c["term_id"]} | LLT: {c["term"]} | PT: {c["pt"]} | SOC: {c.get("soc") or "-"}')
        return "\n".join(lines)

    @classmethod
    async def check_health(cls, force: bool = False) -> dict:
        """Cheaply verify the AI API is reachable and the key is valid.

        Tries GET /models first (usually free, no generation cost). If the
        provider doesn't support that, falls back to a 1-token chat completion.
        Result is cached briefly so the UI can poll this often.
        """
        global _health_cache, _health_cache_at

        if not settings.AI_API_KEY:
            return {"configured": False, "live": False, "detail": "AI_API_KEY not set"}

        now = time.monotonic()
        if not force and _health_cache is not None and (now - _health_cache_at) < HEALTH_CACHE_TTL:
            return _health_cache

        result = await cls._check_models_endpoint()
        if result is None:
            result = await cls._check_via_chat_ping()

        _health_cache, _health_cache_at = result, now
        return result

    @classmethod
    async def _check_models_endpoint(cls) -> dict | None:
        url = MODELS_URL_TMPL.format(base=settings.AI_API_BASE_URL)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=_headers())
        except Exception:
            return None  # fall back to chat ping
        if resp.status_code == 200:
            return {"configured": True, "live": True, "status_code": 200, "checked_via": "models"}
        if resp.status_code == 404:
            return None  # endpoint not supported by this provider — try chat ping
        return {
            "configured": True, "live": False, "status_code": resp.status_code,
            "detail": _classify_error(resp.status_code, resp.text[:200]),
            "checked_via": "models",
        }

    @classmethod
    async def _check_via_chat_ping(cls) -> dict:
        url = CHAT_URL_TMPL.format(base=settings.AI_API_BASE_URL)
        body = {
            "model": settings.AI_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json=body, headers=_headers())
        except Exception as e:
            return {"configured": True, "live": False, "detail": f"AI unreachable — {e}", "checked_via": "chat"}
        if resp.status_code == 200:
            return {"configured": True, "live": True, "status_code": 200, "checked_via": "chat"}
        return {
            "configured": True, "live": False, "status_code": resp.status_code,
            "detail": _classify_error(resp.status_code, resp.text[:200]),
            "checked_via": "chat",
        }

    @classmethod
    async def refine(cls, query: str, candidates: list[dict]) -> dict | None:
        """Call the AI API to interpret + re-rank. Returns parsed dict or None on failure.
        Results are cached by query, so identical queries return instantly."""
        if not settings.AI_API_KEY:
            return None

        # Cache key: query + first 3 candidate IDs (so different candidate sets don't collide)
        cache_key = query.lower() + "|" + "|".join(str(c["term_id"]) for c in candidates[:3])
        if cache_key in _refine_cache:
            return _refine_cache[cache_key]

        prompt = (
            f"USER: {query!r}\nCANDIDATES:\n{cls._candidate_block(candidates)}\n"
            "Return JSON."
        )
        body = {
            "model": settings.AI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 256,
        }
        url = CHAT_URL_TMPL.format(base=settings.AI_API_BASE_URL)
        # Connecting is fast; GENERATING is the slow part for a reasoning model
        # like Kimi K2.6, especially via an "auto" router — give read a lot of room.
        timeout = httpx.Timeout(connect=8.0, read=20.0, write=8.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body, headers=_headers())
            if resp.status_code != 200:
                logger.warning(f"AI API HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            data = resp.json()
            text = _extract_text(data).strip()
            # Strip markdown code fences some providers add despite instructions.
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            if not text:
                logger.warning(f"AI API returned empty content; raw response: {json.dumps(data)[:400]}")
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Salvage the JSON object if there's any surrounding text.
                a, b = text.find("{"), text.rfind("}")
                if a == -1 or b == -1:
                    raise
                parsed = json.loads(text[a:b + 1])
            # Normalize / validate shape.
            result = {
                "interpretation": str(parsed.get("interpretation", "")).strip(),
                "ranked_term_ids": [int(x) for x in parsed.get("ranked_term_ids", []) if str(x).lstrip("-").isdigit()],
                "need_more": bool(parsed.get("need_more", False)),
                "suggested_terms": [str(t).strip() for t in parsed.get("suggested_terms", []) if str(t).strip()][:3],
                "reason": str(parsed.get("reason", "")).strip(),
            }
            # Cache the result
            _refine_cache[cache_key] = result
            if len(_refine_cache) > _refine_cache_max_size:
                # Simple eviction: remove first (oldest) entry
                _refine_cache.pop(next(iter(_refine_cache)))
            return result
        except httpx.TimeoutException as e:
            logger.error(f"AI refine timed out ({type(e).__name__}) — the AI provider took too long to respond.")
            _refine_cache[cache_key] = None  # cache the failure
            return None
        except Exception as e:
            logger.error(f"AI refine failed: {type(e).__name__}: {e!r}")
            _refine_cache[cache_key] = None  # cache the failure
            return None


async def ai_search(engine, query: str, top_k: int) -> dict:
    """
    AI-assisted coding: hybrid search → AI re-rank/interpret → (optional)
    expanded search on the AI's suggested clinical terms. Grounded end-to-end.

    Returns {"query", "count", "results", "ai": {...}}. Falls back to plain hybrid
    search if the AI layer is disabled or unavailable.
    """
    base = engine.search(query, top_k=max(top_k, 15))

    refined = await AIService.refine(query, base[:10]) if base else None
    if not refined:
        return {"query": query, "count": len(base[:top_k]),
                "results": base[:top_k], "ai": {"used": False}}

    pool = {c["term_id"]: c for c in base}

    # If candidates don't capture the meaning, search the AI's clinical terms and
    # merge — our engine is very accurate on clean clinical vocabulary.
    expanded_terms = []
    if refined["need_more"] and refined["suggested_terms"]:
        for term in refined["suggested_terms"]:
            expanded_terms.append(term)
            for c in engine.search(term, top_k=6):
                pool.setdefault(c["term_id"], c)

    # Build final ordering: AI's explicit picks first (grounded to real records),
    # then the rest of the pool by hybrid confidence.
    ordered, seen = [], set()
    for tid in refined["ranked_term_ids"]:
        rec = pool.get(tid) or _record_from_engine(engine, tid)
        if rec and tid not in seen:
            r = dict(rec)
            r["ai_pick"] = True
            ordered.append(r)
            seen.add(tid)

    rest = sorted((c for tid, c in pool.items() if tid not in seen),
                  key=lambda c: c["confidence"], reverse=True)
    ordered.extend(rest)

    # De-duplicate by PT so the list stays clean.
    final, seen_pt = [], set()
    for r in ordered:
        key = (r.get("pt") or r["term"]).lower()
        if key in seen_pt:
            continue
        seen_pt.add(key)
        final.append(r)
        if len(final) >= top_k:
            break

    return {
        "query": query,
        "count": len(final),
        "results": final,
        "ai": {
            "used": True,
            "interpretation": refined["interpretation"],
            "reason": refined["reason"],
            "expanded_terms": expanded_terms,
        },
    }


def _record_from_engine(engine, tid: int):
    t = engine.by_id.get(tid)
    if not t:
        return None
    return {
        "term": t["llt"], "pt": t.get("pt") or t["llt"], "soc": t.get("soc"),
        "term_id": t["id"], "level": t.get("level", "LLT"),
        "confidence": 0.0, "semantic_score": 0.0, "lexical_score": 0.0,
        "match_type": "ai",
    }
