"""
IME (Important Medical Event) index — fast lookups for MedDRA codes/PTs in the
official IME list (MedDRA v29.0).

Loads the Excel once at startup, builds in-memory O(1) lookup dicts by:
  • MedDRA code (term_id)
  • PT name (lowercase for fuzzy match tolerance)

All results are cached and thread-safe.
"""
import logging
import os
import threading

import pandas as pd

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MEDDRA_DATA_DIR", os.path.join(HERE, "..", "data"))
IME_PATH = os.path.join(DATA_DIR, "29-0_ime_list-en.xlsx")


class IMEIndex:
    """In-memory IME list index."""

    def __init__(self):
        self.by_code: dict[int, dict] = {}      # {term_id -> {pt, soc, comment, ...}}
        self.by_pt: dict[str, dict] = {}        # {pt_lower -> same dict}
        self._loaded = False
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load IME Excel sheet into memory."""
        if not os.path.exists(IME_PATH):
            logger.warning(f"IME file not found at {IME_PATH} — IME matching disabled.")
            self._loaded = False
            return

        try:
            df = pd.read_excel(IME_PATH, sheet_name=0, header=11)
            df = df.dropna(subset=["MedDRA Code"])
            logger.info(f"Loading IME list from {IME_PATH}")

            for _, row in df.iterrows():
                try:
                    code = int(row["MedDRA Code"])
                    pt = str(row["PT Name"]).strip()
                    soc = str(row["SOC Name"]).strip() if pd.notna(row["SOC Name"]) else None
                    comment = str(row["Comment"]).strip() if pd.notna(row["Comment"]) else None

                    rec = {
                        "code": code,
                        "pt": pt,
                        "soc": soc,
                        "comment": comment,
                        "is_new": pd.notna(row.get("Added in 29.0")),
                    }

                    self.by_code[code] = rec
                    self.by_pt[pt.lower()] = rec
                except Exception as e:
                    logger.debug(f"IME row parse error: {e}")
                    continue

            logger.info(f"Loaded {len(self.by_code)} IME terms")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load IME index: {e}")
            self._loaded = False

    def lookup_code(self, term_id: int) -> dict | None:
        """Look up IME entry by MedDRA code. Returns dict with pt/soc/comment or None."""
        if not self._loaded:
            return None
        return self.by_code.get(term_id)

    def lookup_pt(self, pt_name: str) -> dict | None:
        """Look up IME entry by PT name (case-insensitive). Returns dict or None."""
        if not self._loaded:
            return None
        return self.by_pt.get((pt_name or "").lower())

    def is_ime(self, term_id: int, pt_name: str = None) -> bool:
        """Quick check: is this term_id or PT in the IME list?"""
        if not self._loaded:
            return False
        if self.lookup_code(term_id):
            return True
        if pt_name and self.lookup_pt(pt_name):
            return True
        return False


# Module-level singleton
_ime_index: IMEIndex | None = None
_lock = threading.Lock()


def get_ime_index() -> IMEIndex:
    """Get or initialize the IME index (thread-safe singleton)."""
    global _ime_index
    if _ime_index is None:
        with _lock:
            if _ime_index is None:
                _ime_index = IMEIndex()
    return _ime_index
