# 2026-01-30 16:33:33 America/New_York
# Purpose: Recompute brought-forward / carried-forward chains across the McD probate OCR JSON corpus,
#          extract page totals + line items, and output an audit table flagging drift and likely causes.



#need to check results

from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# CONFIGURATION
# ----------------------------

COMBINED_FILES = [
    "combined.txt",
]

OUTPUT_DIR = "audit_output"

TOLERANCE = 0.02  # Tolerance for BF~=prev CF threading


# ----------------------------
# Amount parsing
# ----------------------------

UNICODE_FRACTIONS: Dict[str, Fraction] = {
    "¼": Fraction(1,4),
    "½": Fraction(1,2),
    "¾": Fraction(3,4),
    "⅐": Fraction(1,7),
    "⅑": Fraction(1,9),
    "⅒": Fraction(1,10),
    "⅓": Fraction(1,3),
    "⅔": Fraction(2,3),
    "⅕": Fraction(1,5),
    "⅖": Fraction(2,5),
    "⅗": Fraction(3,5),
    "⅘": Fraction(4,5),
    "⅙": Fraction(1,6),
    "⅚": Fraction(5,6),
    "⅛": Fraction(1,8),
    "⅜": Fraction(3,8),
    "⅝": Fraction(5,8),
    "⅞": Fraction(7,8),
}

AMOUNT_TOKEN_RE = re.compile(r"""
(?P<prefix>[$£₣fF]\.?\s*)?              # optional currency
(?P<sign>[-+])?                         # optional sign
(?P<paren_open>\()?\s*                  # optional opening paren
(?P<num>
    (?:\d{1,3}(?:[,\s]\d{3})+|\d+)      # int with thousands or plain
    (?:\.\d{1,2})?                      # optional decimals
    |
    (?:\d{1,3}(?:\.\d{3})+\.\d{1,2})    # OCR artifact like 63.860.08
)
(?:\s*(?P<frac_ascii>\d+\s*/\s*\d+))?   # optional ascii fraction
(?P<frac_uni>[¼½¾⅓⅔⅛⅜⅝⅞⅕⅖⅗⅘⅙⅚⅐⅑⅒])?    # optional unicode fraction
\s*(?P<paren_close>\))?                 # optional closing paren
""", re.VERBOSE)

BF_RE = re.compile(r"amount\s+brought\s+forward", re.IGNORECASE)
CF_RE = re.compile(r"amount\s+carried\s+forward", re.IGNORECASE)
NOTE_NO_RE = re.compile(r"\bNo\.?\s*(\d{1,6})\b", re.IGNORECASE)
APPRAISED_RE = re.compile(r"\bapprais(ed|ement)?\b", re.IGNORECASE)
CONTINUED_RE = re.compile(r"\bcontin(ued|ues|uation)?\b", re.IGNORECASE)
TOTAL_LINE_RE = re.compile(
    r"\b(total\s+amount|making\s+a\s+total\s+amount|total\s+amount\s+of|total\s+of\s+the)\b",
    re.IGNORECASE,
)
SECTION_HEADER_RE = re.compile(
    r"\b(book\s+debts|bills\s+receivable|real\s+estate|inventory|description\s+no\.?|appraised)\b",
    re.IGNORECASE,
)
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
PAGE_WORD_RE = re.compile(r"\b(page|pg\.?|p\.)\b", re.IGNORECASE)


def classify_page_type(text: str) -> str:
    t = text.lower()
    inventory_hits = 0
    ledger_hits = 0
    for kw in (
        "appraised",
        "description no.",
        "inventory",
        "real estate",
        "municipality",
        "tract of land",
        "square of ground",
        "lot of ground",
        "bounded by",
        "front on",
        "depth",
        "parish",
        "plantation",
        "arpent",
        "survey",
    ):
        if kw in t:
            inventory_hits += 1
    for kw in ("ledger", "book debts", "bills receivable"):
        if kw in t:
            ledger_hits += 1
    if inventory_hits >= ledger_hits and inventory_hits > 0:
        return "inventory"
    if ledger_hits > 0:
        return "ledger"
    return "unknown"


def normalize_ocr_number(num_str: str) -> str:
    s = num_str.strip().replace(" ", "")
    # OCR artifact: 63.860.08 => 63860.08
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+\.\d{1,2}", s):
        parts = s.split(".")
        dec = parts[-1]
        whole = "".join(parts[:-1])
        s = f"{whole}.{dec}"
    s = s.replace(",", "")
    return s


def parse_amount_match(m: re.Match) -> float:
    num_raw = m.group("num")
    frac_ascii = m.group("frac_ascii")
    frac_uni = m.group("frac_uni")
    sign = m.group("sign")

    num = float(normalize_ocr_number(num_raw))
    frac = Fraction(0, 1)
    if frac_ascii:
        frac += Fraction(frac_ascii.replace(" ", ""))
    if frac_uni:
        frac += UNICODE_FRACTIONS.get(frac_uni, Fraction(0, 1))

    # Ledger convention: fractions here are fractions-of-a-cent appended after cents.
    num += float(frac) / 100.0
    if sign == "-":
        num = -num
    return num


def is_probable_non_amount(line: str, m: re.Match) -> bool:
    raw = m.group("num")
    if not raw:
        return False
    # If it clearly looks like money (has cents or thousands), keep it
    if re.search(r"\.\d{2}$", raw) or "," in raw:
        return False
    if m.group("prefix"):
        return False

    s = normalize_ocr_number(raw)
    if re.fullmatch(r"\d{4}", s):
        year = int(s)
        if 1700 <= year <= 1899:
            return True

    # If preceded by page/pg/p. within a short window, treat as reference
    pre = line[: m.start()].lower()
    if PAGE_WORD_RE.search(pre[-10:]):
        return True

    # If line contains date language and this is a bare day number, treat as non-amount
    if "day" in line.lower() and MONTH_RE.search(line):
        if re.fullmatch(r"\d{1,2}", s):
            return True

    return False


def extract_totals(text: str) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[str],
    Optional[str],
    bool,
    bool,
    int,
    int,
]:
    bf = cf = None
    bf_raw = cf_raw = None
    bf_incomplete = cf_incomplete = False
    bf_count = 0
    cf_count = 0

    for kind, rex in (("bf", BF_RE), ("cf", CF_RE)):
        matches = list(rex.finditer(text))
        if kind == "bf":
            bf_count = len(matches)
        else:
            cf_count = len(matches)
        if not matches:
            continue
        m = matches[0]
        window = text[m.end(): m.end() + 250]
        am = AMOUNT_TOKEN_RE.search(window)
        if not am:
            if kind == "bf":
                bf_incomplete = True
            else:
                cf_incomplete = True
            continue
        val = parse_amount_match(am)
        raw = window[am.start(): am.end()]
        if kind == "bf":
            bf = val
            bf_raw = raw
        else:
            cf = val
            cf_raw = raw

    return bf, cf, bf_raw, cf_raw, bf_incomplete, cf_incomplete, bf_count, cf_count


# ----------------------------
# Spelled-out amount parsing (heuristic)
# ----------------------------

_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE_WORDS = {"hundred": 100, "thousand": 1000, "million": 1000000}


def _words_to_int(words: List[str]) -> Optional[int]:
    total = 0
    current = 0
    saw = False
    for w in words:
        if w in _NUM_WORDS:
            current += _NUM_WORDS[w]
            saw = True
        elif w in _SCALE_WORDS:
            if current == 0:
                current = 1
            current *= _SCALE_WORDS[w]
            if w in ("thousand", "million"):
                total += current
                current = 0
            saw = True
        elif w == "and":
            continue
        else:
            # Unknown token in number phrase
            return None
    return total + current if saw else None


SPELLED_AMOUNT_RE = re.compile(
    r"(?P<amount>[a-z\\-\\s]+?)\\s+dollars?(?:\\s+and\\s+(?P<cents>[a-z\\-\\s]+?)\\s+cents?)?",
    re.IGNORECASE,
)


def parse_spelled_amount(text: str) -> Optional[float]:
    m = SPELLED_AMOUNT_RE.search(text)
    if not m:
        return None
    amt_words = m.group("amount")
    cents_words = m.group("cents")
    amt_tokens = [t for t in re.split(r"\\s+", amt_words.lower().replace("-", " ")) if t]
    cents_tokens = []
    if cents_words:
        cents_tokens = [t for t in re.split(r"\\s+", cents_words.lower().replace("-", " ")) if t]
    dollars = _words_to_int(amt_tokens)
    cents = _words_to_int(cents_tokens) if cents_tokens else 0
    if dollars is None or cents is None:
        return None
    return float(dollars) + (float(cents) / 100.0)


# ----------------------------
# Reading combined_*.txt (your current format)
# ----------------------------

SPLIT_RE = re.compile(r"\n===== (image\d+\.json) =====\n")


def iter_pages_from_combined(path: str) -> Iterator[Tuple[str, dict]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    parts = SPLIT_RE.split(content)
    for i in range(1, len(parts), 2):
        name = parts[i]
        jtxt = parts[i + 1].strip()
        if not jtxt:
            continue
        yield name, json.loads(jtxt)


def get_page_num(filename: str) -> Optional[int]:
    m = re.search(r"image(\d+)", filename)
    return int(m.group(1)) if m else None


def safe_json_dumps(x) -> str:
    return json.dumps(x or {}, ensure_ascii=False)


def parse_amount_text(text: str) -> Optional[float]:
    if not text:
        return None
    m = AMOUNT_TOKEN_RE.search(str(text))
    if not m:
        return None
    return parse_amount_match(m)


def ocr_snippet(text: str, max_chars: int = 600) -> str:
    if not text:
        return ""
    t = str(text).replace("\r", "")
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def build_category_tables(pages_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    for _, row in pages_df.iterrows():
        page_num = row["page_num"]
        filename = row["filename"]
        try:
            semantic_tags = json.loads(row["semantic_tags"]) if row.get("semantic_tags") else {}
        except Exception:
            semantic_tags = {}

        # Asset-like semantic tags
        for field in ("enslaved_people", "land_assets", "buildings_assets", "livestock_or_equipment"):
            items = semantic_tags.get(field) or []
            for item in items:
                amount = parse_amount_text(item.get("value")) or parse_amount_text(item.get("appraised_value"))
                if amount is None:
                    continue
                rows.append({
                    "page_num": page_num,
                    "filename": filename,
                    "category": field,
                    "amount": amount,
                    "source": "semantic_tags",
                    "field": "value",
                    "party": item.get("name") or item.get("tract_name") or "",
                    "description": item.get("description") or item.get("location") or "",
                    "derived_category": field,
                })

        # Transactions
        txs = semantic_tags.get("transactions") or []
        for tx in txs:
            # Pull all numeric fields so taxonomy can emerge from existing keys
            for key in ("amount_total", "promissory_notes_amount", "book_debt_amount"):
                amount = parse_amount_text(tx.get(key))
                if amount is None:
                    continue
                desc = (tx.get("description") or "").lower()
                derived = None
                if "bills receivable" in desc or "promissory notes" in desc:
                    derived = "bills_receivable"
                elif "book debt" in desc:
                    derived = "book_debt"
                rows.append({
                    "page_num": page_num,
                    "filename": filename,
                    "category": f"transactions.{key}",
                    "amount": amount,
                    "source": "semantic_tags",
                    "field": key,
                    "party": tx.get("party") or "",
                    "description": tx.get("description") or "",
                    "derived_category": derived or f"transactions.{key}",
                })

    categories_df = pd.DataFrame(rows)
    if categories_df.empty:
        return categories_df, categories_df

    agg = (
        categories_df.groupby("derived_category", dropna=False)["amount"]
        .sum()
        .rename("amount_sum")
        .reset_index()
        .sort_values("amount_sum", ascending=False)
        .reset_index(drop=True)
    )
    return categories_df, agg


def build_assets_table(pages_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in pages_df.iterrows():
        page_num = row["page_num"]
        filename = row["filename"]
        raw_ocr = ocr_snippet(row.get("raw_ocr") or "")
        corrected_ocr = ocr_snippet(row.get("corrected_ocr") or "")
        try:
            semantic_tags = json.loads(row["semantic_tags"]) if row.get("semantic_tags") else {}
        except Exception:
            semantic_tags = {}

        # Enslaved people
        for item in semantic_tags.get("enslaved_people") or []:
            rows.append({
                "page_num": page_num,
                "filename": filename,
                "asset_name": item.get("name") or "",
                "asset_type": "enslaved_people",
                "asset_subtype": "",
                "value": parse_amount_text(item.get("appraised_value") or item.get("value")),
                "details": item.get("description") or "",
                "location": item.get("location") or "",
                "age": item.get("age") or "",
                "raw_ocr_context": raw_ocr,
                "corrected_ocr_context": corrected_ocr,
            })

        # Land assets
        for item in semantic_tags.get("land_assets") or []:
            rows.append({
                "page_num": page_num,
                "filename": filename,
                "asset_name": item.get("tract_name") or "",
                "asset_type": "land_assets",
                "asset_subtype": "",
                "value": parse_amount_text(item.get("value")),
                "details": item.get("description") or "",
                "location": item.get("location") or "",
                "acreage": item.get("acreage") or "",
                "raw_ocr_context": raw_ocr,
                "corrected_ocr_context": corrected_ocr,
            })

        # Buildings
        for item in semantic_tags.get("buildings_assets") or []:
            rows.append({
                "page_num": page_num,
                "filename": filename,
                "asset_name": item.get("name") or "",
                "asset_type": "buildings_assets",
                "asset_subtype": "",
                "value": parse_amount_text(item.get("value")),
                "details": item.get("description") or "",
                "location": item.get("location") or "",
                "raw_ocr_context": raw_ocr,
                "corrected_ocr_context": corrected_ocr,
            })

        # Livestock / equipment
        for item in semantic_tags.get("livestock_or_equipment") or []:
            rows.append({
                "page_num": page_num,
                "filename": filename,
                "asset_name": item.get("name") or "",
                "asset_type": "livestock_or_equipment",
                "asset_subtype": "",
                "value": parse_amount_text(item.get("value")),
                "details": item.get("description") or "",
                "location": item.get("location") or "",
                "raw_ocr_context": raw_ocr,
                "corrected_ocr_context": corrected_ocr,
            })

        # Transactions as assets
        for tx in semantic_tags.get("transactions") or []:
            for key in ("amount_total", "promissory_notes_amount", "book_debt_amount"):
                amount = parse_amount_text(tx.get(key))
                if amount is None:
                    continue
                rows.append({
                    "page_num": page_num,
                    "filename": filename,
                    "asset_name": tx.get("party") or "",
                    "asset_type": "transaction",
                    "asset_subtype": key,
                    "value": amount,
                    "details": tx.get("description") or "",
                    "location": "",
                    "age": "",
                    "raw_ocr_context": raw_ocr,
                    "corrected_ocr_context": corrected_ocr,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "asset_id", range(1, len(df) + 1))
    return df


# ----------------------------
# Pipeline
# ----------------------------

def build_tables(combined_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pages: List[dict] = []
    for p in combined_paths:
        for json_name, obj in iter_pages_from_combined(p):
            fname = obj.get("filename") or json_name.replace(".json", ".jpg")
            pages.append({
                "json_name": json_name,
                "filename": fname,
                "page_num": get_page_num(fname),
                "processing_timestamp": obj.get("processing_timestamp"),
                "raw_ocr": obj.get("raw_ocr", ""),
                "corrected_ocr": obj.get("corrected_ocr", ""),
                "categories": safe_json_dumps(obj.get("categories")),
                "semantic_tags": safe_json_dumps(obj.get("semantic_tags")),
                "quantitative_extraction": safe_json_dumps(obj.get("quantitative_extraction")),
                "page_type": classify_page_type(obj.get("corrected_ocr", "") or ""),
                "has_section_header": bool(SECTION_HEADER_RE.search(obj.get("corrected_ocr", "") or "")),
            })

    pages_df = pd.DataFrame(pages).sort_values("page_num").reset_index(drop=True)

    # page totals + line items
    page_totals: List[dict] = []
    line_items: List[dict] = []

    for _, row in pages_df.iterrows():
        text = row["corrected_ocr"] or ""
        bf, cf, bf_raw, cf_raw, bf_inc, cf_inc, bf_count, cf_count = extract_totals(text)
        page_totals.append({
            "page_num": row["page_num"],
            "filename": row["filename"],
            "bf": bf,
            "cf": cf,
            "bf_raw": bf_raw,
            "cf_raw": cf_raw,
            "bf_incomplete": bf_inc,
            "cf_incomplete": cf_inc,
            "bf_count": bf_count,
            "cf_count": cf_count,
        })

        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
        for line_idx, ln in enumerate(lines):
            is_bf = bool(BF_RE.search(ln))
            is_cf = bool(CF_RE.search(ln))
            is_section_total = bool(TOTAL_LINE_RE.search(ln))
            note_m = NOTE_NO_RE.search(ln)
            note_no = int(note_m.group(1)) if note_m else None

            spelled_amount = parse_spelled_amount(ln)
            for am in AMOUNT_TOKEN_RE.finditer(ln):
                if is_probable_non_amount(ln, am):
                    continue
                val = parse_amount_match(am)
                raw = ln[am.start(): am.end()]
                if is_bf:
                    role = "total_bf"
                elif is_cf:
                    role = "total_cf"
                elif is_section_total:
                    role = "section_total"
                else:
                    role = "item"
                spell_delta = None
                spell_mismatch = False
                if spelled_amount is not None:
                    spell_delta = val - spelled_amount
                    if abs(spell_delta) > TOLERANCE:
                        spell_mismatch = True

                line_items.append({
                    "page_num": row["page_num"],
                    "filename": row["filename"],
                    "line_idx": line_idx,
                    "note_no": note_no,
                    "raw_amount": raw,
                    "amount": val,
                    "role": role,
                    "is_continued": bool(CONTINUED_RE.search(ln)),
                    "is_appraised": bool(APPRAISED_RE.search(ln)),
                    "is_section_total": is_section_total,
                    "spelled_amount": spelled_amount,
                    "spelled_mismatch": spell_mismatch,
                    "spelled_delta": spell_delta,
                    "context": ln[:400],
                })

    totals_df = pd.DataFrame(page_totals).sort_values("page_num").reset_index(drop=True)
    line_df = pd.DataFrame(line_items).sort_values(["page_num", "line_idx"]).reset_index(drop=True)
    if not line_df.empty:
        line_counts = line_df.groupby(["page_num", "line_idx"]).size().rename("amounts_on_line").reset_index()
        line_df = line_df.merge(line_counts, on=["page_num", "line_idx"], how="left")
        line_df["multi_amount_line"] = line_df["amounts_on_line"].fillna(0).astype(int) > 1
    return pages_df, totals_df, line_df


def assign_threads(totals_df: pd.DataFrame, tol: float = TOLERANCE) -> pd.DataFrame:
    """Greedy threading: continue thread when BF on page N matches CF on page N-1 within tolerance."""
    totals_df = totals_df.sort_values("page_num").reset_index(drop=True)
    thread_id: List[int] = []
    cur = 0
    prev_cf: Optional[float] = None

    for _, r in totals_df.iterrows():
        bf = r["bf"]
        if prev_cf is not None and pd.notna(bf) and abs(float(bf) - float(prev_cf)) <= tol:
            # continue
            pass
        else:
            cur += 1
        thread_id.append(cur)

        if pd.notna(r["cf"]):
            prev_cf = float(r["cf"])
        else:
            prev_cf = None

    out = totals_df.copy()
    out["thread_id"] = thread_id
    return out


def recompute_audit(
    totals_df: pd.DataFrame,
    line_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    tol: float = TOLERANCE,
) -> pd.DataFrame:
    totals_threaded = assign_threads(totals_df, tol=tol)
    thread_map = totals_threaded.set_index("page_num")["thread_id"].to_dict()

    items = line_df[line_df["role"] == "item"].copy()
    items["thread_id"] = items["page_num"].map(thread_map)
    if not line_df.empty:
        line_df["is_negative"] = line_df["amount"].astype(float) < 0
        line_df["is_zero"] = line_df["amount"].astype(float) == 0

    # Heuristic continuation de-dup:
    # If the last note on page N-1 equals a note on page N and either page N-1 ends with "continued"
    # or the line itself says "continued", skip it on page N.
    continued_by_page: Dict[Tuple[int, int], bool] = {}
    prev_page_note: Dict[Tuple[int, int], Optional[int]] = {}
    prev_page_cont: Dict[Tuple[int, int], bool] = {}

    for (tid, pn), grp in items.groupby(["thread_id", "page_num"]):
        g = grp.sort_values("line_idx")
        tail = g.tail(5)
        continued_by_page[(int(tid), int(pn))] = bool(tail["is_continued"].any())
        nn = g["note_no"].dropna()
        prev_page_note[(int(tid), int(pn))] = int(nn.iloc[-1]) if len(nn) else None
        prev_page_cont[(int(tid), int(pn))] = continued_by_page[(int(tid), int(pn))]

    items = items.sort_values(["thread_id", "page_num", "line_idx"]).reset_index(drop=True)
    items["skip"] = False
    for i, r in items.iterrows():
        tid = int(r["thread_id"]) if pd.notna(r["thread_id"]) else None
        pn = int(r["page_num"])
        note = r["note_no"]
        if tid is None or pd.isna(note):
            continue
        prev_note = prev_page_note.get((tid, pn - 1))
        prev_cont = prev_page_cont.get((tid, pn - 1), False)
        if prev_note is not None and int(note) == int(prev_note) and (bool(r["is_continued"]) or prev_cont):
            items.at[i, "skip"] = True

    items_kept = items[~items["skip"]].copy()
    page_item_sum = items_kept.groupby("page_num")["amount"].sum().rename("item_sum").reset_index()

    audit = totals_threaded.merge(page_item_sum, on="page_num", how="left")
    page_types = pages_df[["page_num", "page_type", "has_section_header"]].drop_duplicates()
    audit = audit.merge(page_types, on="page_num", how="left")
    audit["item_sum"] = audit["item_sum"].fillna(0.0)
    audit["expected_cf"] = np.where(audit["bf"].notna(), audit["bf"] + audit["item_sum"], np.nan)
    audit["delta"] = audit["cf"] - audit["expected_cf"]
    audit["abs_delta"] = audit["delta"].abs()

    # ----------------------------
    # Classify anomalies
    # ----------------------------
    # Page-level signals from extracted line items
    cont_flag = items.groupby("page_num")["is_continued"].any().rename("has_continued").reset_index()
    skipped_flag = items.groupby("page_num")["skip"].any().rename("had_dedup_skip").reset_index()
    spelled_flag = (
        line_df.groupby("page_num")["spelled_mismatch"].any().rename("has_spelled_mismatch").reset_index()
        if not line_df.empty
        else pd.DataFrame(columns=["page_num", "has_spelled_mismatch"])
    )
    multi_amount_flag = (
        line_df.groupby("page_num")["multi_amount_line"].any().rename("has_multi_amount_line").reset_index()
        if not line_df.empty
        else pd.DataFrame(columns=["page_num", "has_multi_amount_line"])
    )
    neg_flag = (
        line_df.groupby("page_num")["is_negative"].any().rename("has_negative_amount").reset_index()
        if not line_df.empty
        else pd.DataFrame(columns=["page_num", "has_negative_amount"])
    )
    zero_flag = (
        line_df.groupby("page_num")["is_zero"].any().rename("has_zero_amount").reset_index()
        if not line_df.empty
        else pd.DataFrame(columns=["page_num", "has_zero_amount"])
    )

    dup_note = pd.DataFrame(columns=["page_num", "has_duplicate_note_no"])
    if not items.empty:
        dup = (
            items.groupby(["page_num", "note_no"]).size().rename("cnt").reset_index()
        )
        dup = dup[dup["note_no"].notna() & (dup["cnt"] > 1)]
        dup_note = dup.groupby("page_num")["cnt"].any().rename("has_duplicate_note_no").reset_index()

    # Does abs(delta) match any single line-item amount on the same page?
    # (Within tolerance, to catch missing/double-count of a single entry.)
    item_amt = items[items["amount"].notna()][["page_num", "amount"]].copy()
    # Merge and compute per-page minimum absolute difference between abs_delta and any item amount
    tmp = audit[["page_num", "abs_delta"]].merge(item_amt, on="page_num", how="left")
    tmp["abs_diff_item"] = (tmp["abs_delta"] - tmp["amount"].abs()).abs()
    min_diff = tmp.groupby("page_num")["abs_diff_item"].min().rename("min_abs_diff_item").reset_index()

    audit = audit.merge(cont_flag, on="page_num", how="left")
    audit = audit.merge(skipped_flag, on="page_num", how="left")
    audit = audit.merge(spelled_flag, on="page_num", how="left")
    audit = audit.merge(multi_amount_flag, on="page_num", how="left")
    audit = audit.merge(neg_flag, on="page_num", how="left")
    audit = audit.merge(zero_flag, on="page_num", how="left")
    audit = audit.merge(dup_note, on="page_num", how="left")
    audit = audit.merge(min_diff, on="page_num", how="left")
    audit["has_continued"] = audit["has_continued"].fillna(False)
    audit["had_dedup_skip"] = audit["had_dedup_skip"].fillna(False)
    audit["has_spelled_mismatch"] = audit["has_spelled_mismatch"].fillna(False)
    audit["has_multi_amount_line"] = audit["has_multi_amount_line"].fillna(False)
    audit["has_negative_amount"] = audit["has_negative_amount"].fillna(False)
    audit["has_zero_amount"] = audit["has_zero_amount"].fillna(False)
    audit["has_duplicate_note_no"] = audit["has_duplicate_note_no"].fillna(False)
    audit["min_abs_diff_item"] = audit["min_abs_diff_item"].fillna(np.inf)

    # Fraction-only drift detection (quarters, halves, thirds, eighths, etc.)
    COMMON_FRACTIONS = [
        0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,
        1.0/3.0, 2.0/3.0,
    ]
    def _is_fraction_only(x: float) -> bool:
        if not np.isfinite(x):
            return False
        ax = abs(x)
        if ax > 1.0 + tol:
            return False
        return any(abs(ax - f) <= max(tol, 0.02) for f in COMMON_FRACTIONS) or ax <= max(tol, 0.02)

    # OCR separator artifact flag
    ocr_sep_pat = re.compile(r"\d+\.\d{3}\.\d{1,2}")
    audit["has_ocr_sep_artifact"] = audit["bf_raw"].fillna("").str.contains(ocr_sep_pat) | audit["cf_raw"].fillna("").str.contains(ocr_sep_pat)
    audit["has_multiple_bf"] = audit["bf_count"].fillna(0).astype(int) > 1
    audit["has_multiple_cf"] = audit["cf_count"].fillna(0).astype(int) > 1

    prev_cf = audit["cf"].shift(1)
    audit["bf_cf_chain_break"] = prev_cf.notna() & audit["bf"].notna() & ((audit["bf"] - prev_cf).abs() > tol)

    audit["anomaly_class"] = "ok"
    audit["audit_eligible"] = (
        (audit["page_type"] == "ledger")
        & (~audit["has_section_header"].fillna(False))
        & audit["bf"].notna()
        & audit["cf"].notna()
    )

    needs = (
        audit["audit_eligible"]
        & (audit["abs_delta"] > tol)
    )

    # Default unexplained for anything non-trivial
    audit.loc[needs, "anomaly_class"] = "unexplained"

    # Incomplete totals override (often page breaks / narrative interruptions)
    inc = needs & (audit["bf_incomplete"] | audit["cf_incomplete"])
    audit.loc[inc, "anomaly_class"] = "incomplete_total"

    # OCR punctuation / thousands separator corruption
    sep = needs & audit["has_ocr_sep_artifact"]
    audit.loc[sep, "anomaly_class"] = "ocr_separator_artifact"

    # Fraction drift (quarters/thirds/eighths)
    frac = needs & audit["delta"].apply(_is_fraction_only)
    audit.loc[frac, "anomaly_class"] = "fraction_drift"

    # Missing/double-count of a single line item
    item_match = needs & (audit["min_abs_diff_item"] <= max(tol, 0.05))
    # If there is explicit continuation or our de-dup kicked in, call it continuation/dedup
    cont = item_match & (audit["has_continued"] | audit["had_dedup_skip"])
    audit.loc[cont, "anomaly_class"] = "continuation_or_dedup"
    # Otherwise, it is likely a missing or double-counted line item
    audit.loc[item_match & ~cont, "anomaly_class"] = "missing_or_doublecount_item"

    # If we have totals but extracted no items (item_sum==0) and no single-item match,
    # this is usually an extraction miss or a page whose arithmetic is not a simple sum-of-items.
    no_items = needs & (audit["item_sum"] == 0.0) & (audit["min_abs_diff_item"] == np.inf)
    audit.loc[no_items, "anomaly_class"] = "no_items_extracted_or_nonadditive_page"

    # If BF or CF is missing, this is a boundary/non-ledger page rather than a ledger arithmetic anomaly
    boundary = audit["bf"].isna() | audit["cf"].isna()
    audit.loc[boundary, "anomaly_class"] = "no_totals_page"

    # Non-ledger pages should not be counted as ledger arithmetic anomalies
    non_ledger = audit["page_type"] == "inventory"
    audit.loc[non_ledger, "anomaly_class"] = "non_ledger_section"

    boundary_section = audit["has_section_header"].fillna(False) & (audit["page_type"] == "ledger")
    audit.loc[boundary_section, "anomaly_class"] = "section_boundary"

    # ----------------------------
    # Thread-level running totals
    # ----------------------------
    audit["thread_expected_cf"] = np.nan
    audit["thread_delta"] = np.nan
    audit["thread_abs_delta"] = np.nan
    audit["thread_discrepancy_reason"] = ""

    for tid, grp in audit.sort_values("page_num").groupby("thread_id"):
        g = grp.copy()
        # Compute running totals across ledger pages (even if not audit-eligible)
        g = g[g["page_type"] == "ledger"].sort_values("page_num")
        if g.empty:
            continue

        running = None
        for idx, row in g.iterrows():
            if bool(row.get("has_section_header")):
                if pd.notna(row["bf"]):
                    running = float(row["bf"]) + float(row["item_sum"])
                else:
                    running = None
                # Even on section headers, record running if we can
            else:
                if running is None:
                    if pd.isna(row["bf"]):
                        continue
                    running = float(row["bf"]) + float(row["item_sum"])
                else:
                    running = running + float(row["item_sum"])

            if running is None:
                continue

            audit.at[idx, "thread_expected_cf"] = running
            if pd.notna(row["cf"]):
                td = float(row["cf"]) - running
                audit.at[idx, "thread_delta"] = td
                audit.at[idx, "thread_abs_delta"] = abs(td)

                if abs(td) <= tol:
                    audit.at[idx, "thread_discrepancy_reason"] = "ok"
                elif row["has_continued"] or row["had_dedup_skip"]:
                    audit.at[idx, "thread_discrepancy_reason"] = "possible_continuation_or_dedup"
                elif row["has_spelled_mismatch"]:
                    audit.at[idx, "thread_discrepancy_reason"] = "spelled_amount_mismatch"
                elif row["min_abs_diff_item"] <= max(tol, 0.05):
                    audit.at[idx, "thread_discrepancy_reason"] = "single_item_mismatch"
                else:
                    audit.at[idx, "thread_discrepancy_reason"] = "unexplained"

    return audit


def write_outputs(
    out_dir: str,
    pages_df: pd.DataFrame,
    totals_df: pd.DataFrame,
    line_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    category_df: pd.DataFrame,
    category_agg_df: pd.DataFrame,
    assets_df: pd.DataFrame,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(df: pd.DataFrame, name: str) -> str:
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        return path

    write_csv(pages_df, "pages.csv")
    write_csv(totals_df, "page_totals.csv")
    write_csv(line_df, "line_items.csv")
    write_csv(audit_df, "ledger_audit.csv")
    if category_df is not None and not category_df.empty:
        write_csv(category_df, "category_items.csv")
    if category_agg_df is not None and not category_agg_df.empty:
        write_csv(category_agg_df, "category_agg.csv")
    if assets_df is not None and not assets_df.empty:
        write_csv(assets_df, "assets.csv")

    worst = audit_df.sort_values("abs_delta", ascending=False).head(500)
    worst.to_csv(os.path.join(out_dir, "worst_deltas.csv"), index=False)

    # Optional parquet if an engine exists
    try:
        pages_df.to_parquet(os.path.join(out_dir, "pages.parquet"), index=False)
        totals_df.to_parquet(os.path.join(out_dir, "page_totals.parquet"), index=False)
        line_df.to_parquet(os.path.join(out_dir, "line_items.parquet"), index=False)
        audit_df.to_parquet(os.path.join(out_dir, "ledger_audit.parquet"), index=False)
        if category_df is not None and not category_df.empty:
            category_df.to_parquet(os.path.join(out_dir, "category_items.parquet"), index=False)
        if category_agg_df is not None and not category_agg_df.empty:
            category_agg_df.to_parquet(os.path.join(out_dir, "category_agg.parquet"), index=False)
        if assets_df is not None and not assets_df.empty:
            assets_df.to_parquet(os.path.join(out_dir, "assets.parquet"), index=False)
    except Exception:
        # No parquet engine installed, CSV outputs are still complete.
        pass

    write_markdown_report(out_dir, pages_df, line_df, audit_df, category_agg_df, assets_df)
    write_error_pages(out_dir, pages_df, audit_df)
    generate_distribution_pngs(out_dir, assets_df)
    generate_pdf_report(out_dir, pages_df, line_df, audit_df, category_agg_df, assets_df)


def _format_number(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.2f}"
    return str(v)


def _df_to_markdown(df: pd.DataFrame, columns: List[str], max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    show = df[columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, r in show.iterrows():
        vals = []
        for c in columns:
            v = r.get(c, "")
            vals.append(_format_number(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    out_dir: str,
    pages_df: pd.DataFrame,
    line_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    category_agg_df: pd.DataFrame,
    assets_df: pd.DataFrame = None,
) -> None:
    path = os.path.join(out_dir, "audit_report.md")
    total_pages = len(pages_df)
    total_lines = len(line_df)
    eligible = audit_df[audit_df["audit_eligible"]] if "audit_eligible" in audit_df.columns else audit_df
    flagged = int((eligible["abs_delta"] > TOLERANCE).sum()) if not eligible.empty else 0
    worst = eligible.sort_values("abs_delta", ascending=False).head(50)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Probate Ledger Audit Report\n\n")
        total_pages_fmt = f"{total_pages:,}"
        total_lines_fmt = f"{total_lines:,}"
        eligible_count = len(eligible)
        eligible_pct = (eligible_count / total_pages * 100.0) if total_pages else 0.0
        flagged_pct = (flagged / eligible_count * 100.0) if eligible_count else 0.0
        f.write(f"- Pages: {total_pages_fmt}\n")
        f.write(f"- Line items extracted: {total_lines_fmt}\n")
        f.write(
            "- Audit-eligible pages: "
            f"{eligible_count:,} ({eligible_pct:.2f}%)\n"
        )
        f.write(
            "- Pages with delta > tolerance "
            f"({TOLERANCE}) among audit-eligible pages: "
            f"{flagged:,} ({flagged_pct:.2f}%)\n\n"
        )

        f.write("## Worst Deltas (Top 50)\n\n")
        f.write(_df_to_markdown(
            worst,
            ["page_num", "bf", "cf", "item_sum", "expected_cf", "delta", "anomaly_class"],
            max_rows=50,
        ))
        f.write("\n")

        f.write("## Error Summary (Audit-Eligible Pages)\n\n")
        err = audit_df[audit_df["audit_eligible"] & (audit_df["anomaly_class"] != "ok")].copy()
        if err.empty:
            f.write("_No audit-eligible pages._\n\n")
        else:
            err_counts = err["anomaly_class"].value_counts().rename_axis("anomaly_class").reset_index(name="count")
            err_counts["percent_of_eligible"] = err_counts["count"] / max(eligible_count, 1) * 100.0
            f.write(_df_to_markdown(err_counts, ["anomaly_class", "count", "percent_of_eligible"], max_rows=50))
            f.write("\n")

            if "thread_discrepancy_reason" in audit_df.columns:
                thread_err = audit_df[
                    audit_df["audit_eligible"]
                    & audit_df["thread_abs_delta"].notna()
                    & (audit_df["thread_abs_delta"] > TOLERANCE)
                ]
                tcounts = (
                    thread_err["thread_discrepancy_reason"]
                    .fillna("")
                    .replace("", "none")
                    .value_counts()
                    .rename_axis("thread_discrepancy_reason")
                    .reset_index(name="count")
                )
                tcounts["percent_of_eligible"] = tcounts["count"] / max(eligible_count, 1) * 100.0
                f.write("### Thread Discrepancy Reasons\n\n")
                f.write(_df_to_markdown(tcounts, ["thread_discrepancy_reason", "count", "percent_of_eligible"], max_rows=50))
                f.write("\n")

        f.write("## Category Aggregates (Top 50)\n\n")
        if category_agg_df is None or category_agg_df.empty:
            f.write("_No category aggregates available._\n")
        else:
            cat = category_agg_df.copy()
            total_amt = cat["amount_sum"].sum()
            cat["percent_of_total"] = cat["amount_sum"] / max(total_amt, 1.0) * 100.0
            f.write(_df_to_markdown(cat, ["derived_category", "amount_sum", "percent_of_total"], max_rows=50))

        f.write("\n")
        f.write("## Asset & Transaction Stats\n\n")
        if assets_df is None or assets_df.empty:
            f.write("_No assets available._\n")
            return

        assets = assets_df.copy()
        assets["value"] = pd.to_numeric(assets["value"], errors="coerce")

        def _stats_table(df: pd.DataFrame, label: str, group_col: str = None) -> None:
            if df.empty:
                f.write(f"_No {label} records._\n\n")
                return
            if group_col is None:
                vals = df["value"].dropna()
                if vals.empty:
                    f.write(f"_No numeric values for {label}._\n\n")
                    return
                stats = pd.DataFrame([{
                    "group": label,
                    "total_records": len(df),
                    "value_count": len(vals),
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "stdev": vals.std(ddof=0),
                }])
                f.write(_df_to_markdown(stats, ["group", "total_records", "value_count", "mean", "median", "stdev"], max_rows=50))
                f.write("\n")
                return

            rows = []
            for key, g in df.groupby(group_col):
                vals = g["value"].dropna()
                if vals.empty:
                    continue
                rows.append({
                    "group": key,
                    "total_records": len(g),
                    "value_count": len(vals),
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "stdev": vals.std(ddof=0),
                })
            if not rows:
                f.write(f"_No numeric values for {label} by {group_col}._\n\n")
                return
            table = pd.DataFrame(rows).sort_values("mean", ascending=False)
            f.write(_df_to_markdown(table, ["group", "total_records", "value_count", "mean", "median", "stdev"], max_rows=50))
            f.write("\n")

        # Overall assets and transactions
        assets_only = assets[assets["asset_type"] != "transaction"]
        tx_only = assets[assets["asset_type"] == "transaction"]
        f.write("### Overall\n\n")
        _stats_table(assets_only, "assets")
        _stats_table(tx_only, "transactions")

        # By subcategory
        f.write("### By Asset Type\n\n")
        _stats_table(assets_only, "assets by type", group_col="asset_type")

        f.write("### By Transaction Subtype\n\n")
        _stats_table(tx_only, "transactions by subtype", group_col="asset_subtype")


def write_error_pages(out_dir: str, pages_df: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    errors_dir = os.path.join(out_dir, "errors")
    os.makedirs(errors_dir, exist_ok=True)

    # Define error conditions broadly (any anomaly or thread discrepancy or notable flags)
    err = audit_df.copy()
    err["has_thread_issue"] = (
        err.get("audit_eligible", False)
        & err.get("thread_abs_delta").notna()
        & (err.get("thread_abs_delta") > TOLERANCE)
    )
    non_error_classes = {"ok", "non_ledger_section", "section_boundary", "no_totals_page"}
    err["has_anomaly"] = ~err["anomaly_class"].fillna("ok").isin(non_error_classes)
    err["has_flags"] = (
        err["has_spelled_mismatch"].fillna(False)
        | err["has_multi_amount_line"].fillna(False)
        | err["has_negative_amount"].fillna(False)
        | err["has_zero_amount"].fillna(False)
        | err["has_duplicate_note_no"].fillna(False)
        | err["has_multiple_bf"].fillna(False)
        | err["has_multiple_cf"].fillna(False)
        | err["bf_cf_chain_break"].fillna(False)
    )
    err = err[err["has_anomaly"] | err["has_thread_issue"] | err["has_flags"]]

    if err.empty:
        return

    # Summary CSV for manual review
    summary_cols = [
        "page_num",
        "filename",
        "page_type",
        "anomaly_class",
        "delta",
        "thread_delta",
        "thread_discrepancy_reason",
        "has_spelled_mismatch",
        "has_multi_amount_line",
        "has_negative_amount",
        "has_zero_amount",
        "has_duplicate_note_no",
        "has_multiple_bf",
        "has_multiple_cf",
        "bf_cf_chain_break",
    ]
    summary_path = os.path.join(errors_dir, "error_summary.csv")
    err[summary_cols].to_csv(summary_path, index=False)

    # Write per-page markdown with error list + full page text
    pages_lookup = pages_df.set_index("page_num")[["filename", "corrected_ocr", "raw_ocr", "page_type"]]
    for _, row in err.iterrows():
        pn = int(row["page_num"])
        if pn not in pages_lookup.index:
            continue
        page = pages_lookup.loc[pn]
        md_path = os.path.join(errors_dir, f"page_{pn:05d}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Page {pn}\n\n")
            f.write(f"- Filename: {page['filename']}\n")
            f.write(f"- Page type: {page['page_type']}\n")
            f.write(f"- Anomaly class: {row.get('anomaly_class')}\n")
            f.write(f"- Delta: {row.get('delta')}\n")
            f.write(f"- Thread delta: {row.get('thread_delta')}\n")
            f.write(f"- Thread reason: {row.get('thread_discrepancy_reason')}\n\n")

            f.write("## Flags\n\n")
            f.write(f"- has_spelled_mismatch: {row.get('has_spelled_mismatch')}\n")
            f.write(f"- has_multi_amount_line: {row.get('has_multi_amount_line')}\n")
            f.write(f"- has_negative_amount: {row.get('has_negative_amount')}\n")
            f.write(f"- has_zero_amount: {row.get('has_zero_amount')}\n")
            f.write(f"- has_duplicate_note_no: {row.get('has_duplicate_note_no')}\n")
            f.write(f"- has_multiple_bf: {row.get('has_multiple_bf')}\n")
            f.write(f"- has_multiple_cf: {row.get('has_multiple_cf')}\n")
            f.write(f"- bf_cf_chain_break: {row.get('bf_cf_chain_break')}\n\n")

            f.write("## Page Text (Corrected OCR)\n\n")
            f.write("```\n")
            f.write(str(page["corrected_ocr"] or "").rstrip() + "\n")
            f.write("```\n")


def generate_distribution_pngs(out_dir: str, assets_df: pd.DataFrame, bins: int = 30) -> None:
    if assets_df is None or assets_df.empty:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        os.makedirs("/tmp/mplconfig", exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    assets = assets_df.copy()
    assets["value"] = pd.to_numeric(assets["value"], errors="coerce")
    assets_only = assets[assets["asset_type"] != "transaction"]
    tx_only = assets[assets["asset_type"] == "transaction"]

    dist_dir = os.path.join(out_dir, "distributions")
    os.makedirs(dist_dir, exist_ok=True)

    def plot_distribution(values, title, path):
        vals = values.dropna()
        if vals.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=bins, color="#4c78a8", edgecolor="white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    for asset_type, g in assets_only.groupby("asset_type"):
        fname = f"asset_{asset_type}.png"
        plot_distribution(g["value"], f"Distribution: {asset_type}", os.path.join(dist_dir, fname))

    for subtype, g in tx_only.groupby("asset_subtype"):
        fname = f"transaction_{subtype}.png"
        plot_distribution(g["value"], f"Distribution: transaction {subtype}", os.path.join(dist_dir, fname))


def generate_pdf_report(
    out_dir: str,
    pages_df: pd.DataFrame,
    line_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    category_agg_df: pd.DataFrame,
    assets_df: pd.DataFrame,
) -> None:
    if assets_df is None or assets_df.empty:
        return
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return

    os.makedirs("/tmp/mplconfig", exist_ok=True)

    def fmt_num(x) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            return f"{float(x):,.2f}"
        return str(x)

    def table_data(df: pd.DataFrame, columns: List[str]) -> List[List[str]]:
        data = []
        for _, r in df.iterrows():
            data.append([fmt_num(r.get(c)) for c in columns])
        return data

    def draw_table(ax, df: pd.DataFrame, columns: List[str], col_widths: List[float]) -> None:
        ax.axis("off")
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9, color="#666666")
            return

        data = table_data(df, columns)
        tbl = ax.table(
            cellText=data,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
            loc="center",
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

        header_color = "#f2f4f8"
        row_colors = ["#ffffff", "#f8fafc"]
        edge_color = "#d0d5dd"

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(edge_color)
            if row == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(weight="bold", color="#111827")
            else:
                cell.set_facecolor(row_colors[(row - 1) % 2])

    num_pages = len(pages_df)
    num_lines = len(line_df)
    eligible = audit_df[audit_df.get("audit_eligible", False) == True]
    flagged = eligible[eligible["abs_delta"] > TOLERANCE]
    eligible_count = len(eligible)
    flagged_count = len(flagged)

    worst = eligible.sort_values("abs_delta", ascending=False).head(4)
    worst_cols = ["page_num", "bf", "cf", "item_sum", "expected_cf", "delta", "anomaly_class"]

    err_counts = (
        eligible[eligible["anomaly_class"] != "ok"]["anomaly_class"]
        .value_counts()
        .reset_index()
    )
    err_counts.columns = ["anomaly_class", "count"]

    cat_top = category_agg_df.sort_values("amount_sum", ascending=False).head(6)

    assets = assets_df.copy()
    assets["value"] = pd.to_numeric(assets["value"], errors="coerce")
    assets_only = assets[assets["asset_type"] != "transaction"]
    tx_only = assets[assets["asset_type"] == "transaction"]

    def stats_row(df, label):
        vals = df["value"].dropna()
        if vals.empty:
            return {"group": label, "total_records": len(df), "value_count": 0, "mean": np.nan, "median": np.nan, "stdev": np.nan}
        return {
            "group": label,
            "total_records": len(df),
            "value_count": len(vals),
            "mean": vals.mean(),
            "median": vals.median(),
            "stdev": vals.std(ddof=0),
        }

    overall_stats = pd.DataFrame([
        stats_row(assets_only, "assets"),
        stats_row(tx_only, "transactions"),
    ])

    asset_type_stats = (
        assets_only.groupby("asset_type")["value"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    asset_type_stats.rename(columns={"asset_type": "group", "std": "stdev", "count": "value_count"}, inplace=True)
    asset_type_stats["total_records"] = asset_type_stats["value_count"]
    asset_type_stats = asset_type_stats[["group", "total_records", "value_count", "mean", "median", "stdev"]].head(6)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"

    out_path = os.path.join(out_dir, "audit_report_with_distributions.pdf")
    with PdfPages(out_path) as pdf:
        # Summary page
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        gs = fig.add_gridspec(6, 2, height_ratios=[0.55, 0.6, 1.2, 1.0, 1.0, 1.4], hspace=0.35, wspace=0.25)

        fig.text(0.5, 0.965, "Probate Audit Report", ha="center", va="top", fontsize=18, fontweight="bold", color="#111827")
        fig.text(0.5, 0.94, "Executive Summary", ha="center", va="top", fontsize=10, color="#6b7280")

        stats_text = (
            f"Pages: {num_pages:,}    |    Line items: {num_lines:,}\n"
            f"Audit-eligible pages: {eligible_count:,} ({eligible_count/num_pages*100 if num_pages else 0:.2f}%)    |    "
            f"Delta > {TOLERANCE}: {flagged_count:,} ({flagged_count/eligible_count*100 if eligible_count else 0:.2f}%)"
        )
        fig.text(0.06, 0.89, stats_text, ha="left", va="top", fontsize=10, color="#111827")

        ax_worst = fig.add_subplot(gs[2, :])
        ax_worst.set_title("Worst Deltas (Top 4)", loc="left", fontsize=11, pad=6)
        draw_table(ax_worst, worst, worst_cols, [0.08, 0.12, 0.12, 0.12, 0.14, 0.12, 0.20])

        ax_err = fig.add_subplot(gs[3, 0])
        ax_err.set_title("Error Summary (Eligible Pages)", loc="left", fontsize=11, pad=6)
        draw_table(ax_err, err_counts, ["anomaly_class", "count"], [0.7, 0.3])

        ax_cat = fig.add_subplot(gs[3, 1])
        ax_cat.set_title("Category Aggregates (Top 6)", loc="left", fontsize=11, pad=6)
        draw_table(ax_cat, cat_top, ["derived_category", "amount_sum"], [0.65, 0.35])

        ax_stats = fig.add_subplot(gs[4, :])
        ax_stats.set_title("Asset & Transaction Stats (Overall)", loc="left", fontsize=11, pad=6)
        draw_table(ax_stats, overall_stats, ["group", "total_records", "value_count", "mean", "median", "stdev"], [0.22, 0.18, 0.18, 0.14, 0.14, 0.14])

        ax_stats2 = fig.add_subplot(gs[5, :])
        ax_stats2.set_title("Asset Stats (By Type, Top 6)", loc="left", fontsize=11, pad=6)
        draw_table(ax_stats2, asset_type_stats, ["group", "total_records", "value_count", "mean", "median", "stdev"], [0.28, 0.16, 0.16, 0.14, 0.14, 0.12])

        fig.text(0.06, 0.03, "Generated from audit_output/*.csv", ha="left", va="bottom", fontsize=8, color="#6b7280")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Distribution pages: asset types
        asset_types = sorted([t for t in assets["asset_type"].unique() if t != "transaction"])
        per_page = 4
        for i in range(0, len(asset_types), per_page):
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
            fig.suptitle("Distributions by Asset Type", fontsize=14, fontweight="bold", y=0.98)
            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")
            for ax_idx, asset_type in enumerate(asset_types[i:i + per_page]):
                ax = axes[ax_idx]
                vals = assets_only[assets_only["asset_type"] == asset_type]["value"].dropna()
                ax.axis("on")
                ax.hist(vals, bins=30, color="#4c78a8", edgecolor="white")
                ax.set_title(asset_type, fontsize=10, fontweight="bold")
                ax.set_xlabel("Value", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.grid(axis="y", alpha=0.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Distribution pages: transaction subtypes
        subtypes = sorted(tx_only["asset_subtype"].dropna().unique().tolist())
        for i in range(0, len(subtypes), per_page):
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
            fig.suptitle("Distributions by Transaction Subtype", fontsize=14, fontweight="bold", y=0.98)
            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")
            for ax_idx, subtype in enumerate(subtypes[i:i + per_page]):
                ax = axes[ax_idx]
                vals = tx_only[tx_only["asset_subtype"] == subtype]["value"].dropna()
                ax.axis("on")
                ax.hist(vals, bins=30, color="#f28e2b", edgecolor="white")
                ax.set_title(subtype, fontsize=10, fontweight="bold")
                ax.set_xlabel("Value", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.grid(axis="y", alpha=0.2)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    pages_df, totals_df, line_df = build_tables(COMBINED_FILES)
    audit_df = recompute_audit(totals_df, line_df, pages_df, tol=TOLERANCE)
    category_df, category_agg_df = build_category_tables(pages_df)
    assets_df = build_assets_table(pages_df)
    write_outputs(OUTPUT_DIR, pages_df, totals_df, line_df, audit_df, category_df, category_agg_df, assets_df)


if __name__ == "__main__":
    main()
