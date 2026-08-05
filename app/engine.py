"""
MedDRA Coding Engine — hybrid fuzzy + semantic search.

Given free-text (a symptom, complaint, abbreviation, or misspelling), it returns
the closest MedDRA LLT/PT terms ranked with a confidence score. Everything runs
in-memory: no OpenSearch, no database, no PyTorch.

Channels combined:
  • Semantic  — ONNX sentence embeddings (fastembed / bge-small), catches meaning
                and paraphrases ("drug not working" -> "Drug ineffective").
  • Lexical   — RapidFuzz, catches spelling mistakes, abbreviations, word order.
  • Exact     — direct/substring hits get a confidence boost.
"""
import gzip
import json
import logging
import os
import threading

import numpy as np
from rapidfuzz import fuzz, process

from app.config import settings

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MEDDRA_DATA_DIR", os.path.join(HERE, "..", "data"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class MeddraEngine:
    """Loads the prebuilt index once and answers coding queries."""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.terms: list[dict] = []
        self.llt_names: list[str] = []
        self.llt_lower: list[str] = []
        self.vectors: np.ndarray | None = None          # float16 [N, dim], L2-normalized
        self._model = None
        self._model_lock = threading.Lock()
        self._load()

    # ── Loading ─────────────────────────────────────────────────────
    def _load(self):
        terms_path = os.path.join(self.data_dir, "meddra_terms.jsonl.gz")
        vec_path = os.path.join(self.data_dir, "meddra_vectors.npz")

        logger.info(f"Loading MedDRA terms from {terms_path}")
        with gzip.open(terms_path, "rt", encoding="utf-8") as f:
            self.terms = [json.loads(line) for line in f]
        self.llt_names = [t["llt"] for t in self.terms]
        self.llt_lower = [n.lower() for n in self.llt_names]
        self.by_id = {t["id"]: t for t in self.terms}
        logger.info(f"Loaded {len(self.terms)} LLT terms")

        if not settings.ENABLE_SEMANTIC:
            self.vectors = None
            logger.info("LITE MODE: semantic search disabled (fuzzy/lexical only).")
            return

        logger.info(f"Loading semantic vectors from {vec_path}")
        npz = np.load(vec_path, allow_pickle=True)
        q = npz["vectors"].astype(np.float32) / 127.0     # dequantize int8 -> float
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        self.vectors = q.astype(np.float16)               # ~45 MB resident
        logger.info(f"Loaded vectors {self.vectors.shape}")

    # ── Query embedding (lazy, thread-safe) ─────────────────────────
    def _get_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from fastembed import TextEmbedding
                    logger.info(f"Loading embedding model: {EMBED_MODEL}")
                    cache_dir = os.environ.get("FASTEMBED_CACHE_DIR")  # baked into image at build
                    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
                    self._model = TextEmbedding(model_name=EMBED_MODEL, **kwargs)
                    logger.info("Embedding model ready")
        return self._model

    def _embed_query(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = np.array(list(model.embed([text]))[0], dtype=np.float32)
        vec /= (np.linalg.norm(vec) + 1e-9)
        return vec

    def warmup(self):
        """Preload the model so the first real query is fast (no-op in lite mode)."""
        if not settings.ENABLE_SEMANTIC:
            return
        try:
            self._embed_query("warmup")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    # ── Search ──────────────────────────────────────────────────────
    def search(self, text: str, top_k: int = 8, sem_candidates: int = 150,
               fuzz_candidates: int = 150) -> list[dict]:
        text = (text or "").strip()
        if not text:
            return []
        q_lower = text.lower()

        # 1) Semantic candidates (skipped entirely in lite mode)
        sem_idx = np.array([], dtype=int)
        sims = None
        if settings.ENABLE_SEMANTIC and self.vectors is not None:
            try:
                qv = self._embed_query(text).astype(np.float16)
                sims = self.vectors @ qv                   # cosine, shape [N]
                sims = sims.astype(np.float32)
                sem_idx = np.argpartition(-sims, sem_candidates)[:sem_candidates]
            except Exception as e:
                logger.error(f"Semantic search failed, using lexical only: {e}")

        # 2) Lexical/fuzzy candidates. Match on the lowercased query against the
        #    lowercased term list (WRatio is case-sensitive; using the raw query
        #    would make capitalized input like "SOB" miss its own exact term).
        fuzzy_hits = process.extract(
            q_lower, self.llt_lower, scorer=fuzz.WRatio,
            limit=fuzz_candidates, score_cutoff=55,
        )
        fuzz_idx = [h[2] for h in fuzzy_hits]

        # 3) Union of candidate rows
        cand = set(int(i) for i in sem_idx.tolist()) | set(int(i) for i in fuzz_idx)
        if not cand:
            return []

        # 4) Score each candidate = blend(semantic, lexical) + exact/substring boost.
        #    Lexical uses token_set_ratio (robust to word order / subsets) and is
        #    damped when the candidate is much shorter than the query, so tiny
        #    generic LLTs ("BP", "Ache", "Cap") can't hijack a longer query.
        qlen = max(len(q_lower), 1)
        scored = []
        for i in cand:
            name_l = self.llt_lower[i]
            sem = float(sims[i]) if sims is not None else 0.0
            sem01 = max(0.0, min(1.0, sem))                # cosine -> [0,1]

            # WRatio blends ratio/partial/token scorers → good for typos AND word
            # subsets. Damp it when the candidate is far shorter than the query so
            # tiny generic LLTs can't hijack a longer phrase.
            # ── Lexical confidence (word overlap, typos, exact/substring) ──
            # token_set_ratio weighs ALL tokens, so a long term that merely
            # *contains* a shared word isn't unfairly boosted. Damp short generic
            # candidates so they can't hijack a longer query.
            len_ratio = len(name_l) / qlen
            length_factor = max(0.45, min(1.0, 0.45 + 0.55 * len_ratio))
            token = (fuzz.token_set_ratio(q_lower, name_l) / 100.0) * length_factor
            ratio = fuzz.ratio(q_lower, name_l) / 100.0       # edit distance (typos)

            exact = 1.0 if q_lower == name_l else 0.0
            substr = 0.0
            if not exact and len(name_l) >= 4 and (q_lower in name_l or name_l in q_lower):
                substr = 0.85 * length_factor
            typo = ratio if ratio >= 0.82 else 0.0            # near-exact spelling

            lex_conf = max(exact, typo, substr, 0.92 * token)

            # ── Combine channels with MAX, not a sum ──
            # A confident lexical hit (typo/exact/abbrev) wins on lex_conf; a
            # confident paraphrase wins on semantics. They no longer cannibalize
            # each other. Small semantic discount so exact ties edge out fuzzy sem.
            rank_score = max(lex_conf, 0.98 * sem01)
            rank_score -= 0.0009 * len(name_l)                # brevity tiebreak (canonical first)

            if lex_conf >= 0.6 and sem01 >= 0.6:
                match_type = "semantic+lexical"
            elif lex_conf > sem01:
                match_type = "lexical"
            else:
                match_type = "semantic"
            scored.append((rank_score, sem01, lex_conf, i, match_type))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 5) Deduplicate by Preferred Term (keep the best-ranked LLT per PT)
        results, seen_pt = [], set()
        for rank_score, sem01, lex_conf, i, match_type in scored:
            t = self.terms[i]
            pt_key = (t.get("pt") or t["llt"]).lower()
            if pt_key in seen_pt:
                continue
            seen_pt.add(pt_key)
            results.append({
                "term": t["llt"],
                "pt": t.get("pt") or t["llt"],
                "soc": t.get("soc"),
                "term_id": t["id"],
                "level": t.get("level", "LLT"),
                "confidence": round(max(0.0, min(1.0, rank_score)) * 100, 1),
                "semantic_score": round(sem01 * 100, 1),
                "lexical_score": round(min(1.0, lex_conf) * 100, 1),
                "match_type": match_type,
            })
            if len(results) >= top_k:
                break
        return results


# Module-level singleton
_engine: MeddraEngine | None = None
_lock = threading.Lock()


def get_engine() -> MeddraEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = MeddraEngine()
    return _engine
