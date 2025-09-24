#!/usr/bin/env python3
"""
batch_all_biotech.py
--------------------
Build US (+ optional ASX) biotech universe and run activity screen.

Key features:
  • Universe: NASDAQ JSON (ticker+name), ASX from local CSV
  • Metadata enrichment (name/sector/industry/market_cap):
        yahooquery (bulk) -> yfinance fallback (only for missing sector/industry)
  • On-disk cache (Parquet) so repeated runs are fast
  • Biotech filter AFTER enrichment
  • Only writes CSV if content changed (idempotent)
  • Compatible with biotech_activity_screen.py (same directory)

Usage examples:
  python batch_all_biotech.py --us
  python batch_all_biotech.py --us --blurbs --openai-key sk-XXXX
  python batch_all_biotech.py --asx --asx-file asx_tickers.csv
  python batch_all_biotech.py --us --asx --asx-file asx_tickers.csv --batch-size 20 --sleep-batches 1.5
"""

from __future__ import annotations
import os, io, sys, time, json, hashlib, argparse
from typing import List, Optional
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

API_KEY = os.getenv("OPENAI_API_KEY")
try:
    from biotech_activity_screen import screen_activity_for_tickers
except Exception as e:
    print("ERROR: could not import biotech_activity_screen.py — place it next to this script.")
    raise

# ---------- Constants ----------
NASDAQ_JSON = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000"
NASDQ_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_ASX_LOCAL = "asx_tickers.csv"
META_CACHE_PATH = "meta_cache.parquet"

INDUSTRY_MATCH = [
    "biotech", "biotechnology", "biotechnology & medical research",
    "life sciences", "pharmaceuticals"
]

# ---------- Small utilities ----------

def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _sha256_df(df: pd.DataFrame) -> str:
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()

def _is_biotech_text(industry: str, sector: str) -> bool:
    s = f"{(industry or '').lower()} {(sector or '').lower()}"
    return any(k in s for k in INDUSTRY_MATCH)

# ---------- Universe builders (NO prefiltering here) ----------

def fetch_nasdaq_universe() -> pd.DataFrame:
    """
    NASDAQ API reliably returns ticker + company; sector/industry are usually empty.
    We take only ticker & company here; biotech filtering happens AFTER enrichment.
    """
    try:
        r = requests.get(NASDAQ_JSON, headers=NASDQ_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = (data or {}).get("data", {}).get("table", {}).get("rows", [])
        if not rows:
            raise RuntimeError("No rows in NASDAQ API response.")
        df = pd.DataFrame(rows)
        if not {"symbol", "name"} <= set(df.columns):
            raise RuntimeError(f"Unexpected NASDAQ columns: {df.columns.tolist()}")
        df = df.rename(columns={"symbol": "ticker", "name": "company"})[["ticker", "company"]]
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["company"] = df["company"].astype(str).str.strip()
        df = df.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)
        return df.rename(columns={"company": "name"})
    except Exception as e:
        print(f"[WARN] NASDAQ fetch failed: {e}")
        if os.path.exists("us_tickers.csv"):
            print("[INFO] Using local fallback: us_tickers.csv")
            df = pd.read_csv("us_tickers.csv")
            need = ["ticker","name"]
            for c in need:
                if c not in df.columns:
                    df[c] = None
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df["name"] = df["name"].astype(str).str.strip()
            return df[need].dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)
        return pd.DataFrame(columns=["ticker","name"])

def fetch_asx_universe(local_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Use a local CSV with a column 'code' or 'ticker' (e.g., OPT, IMU, KZA).
    Adds '.AX' suffix if missing.
    """
    path = local_csv or DEFAULT_ASX_LOCAL
    if not os.path.exists(path):
        raise RuntimeError(
            f"ASX list file not found: {path}. Provide a CSV with a 'code' or 'ticker' column."
        )
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if c.strip().lower() in ("code","ticker","asx code","asx"):
            col = c; break
    if not col:
        raise RuntimeError("ASX CSV must include a 'code' or 'ticker' column.")
    codes = df[col].astype(str).str.upper().str.strip()
    tickers = codes.where(codes.str.endswith(".AX"), codes + ".AX")
    out = pd.DataFrame({"ticker": tickers})
    out["name"] = out["ticker"]
    out = out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)
    return out

def build_universe(include_asx: bool, include_us: bool, asx_file: Optional[str]) -> pd.DataFrame:
    frames = []
    if include_us:
        us = fetch_nasdaq_universe()  # ticker + name only
        frames.append(us[["ticker","name"]])
    if include_asx:
        try:
            asx = fetch_asx_universe(asx_file)
            frames.append(asx[["ticker","name"]])
        except Exception as e:
            print(f"[WARN] ASX fetch failed: {e} — continuing without ASX.")
    if not frames:
        return pd.DataFrame(columns=["ticker","name"])
    uni = pd.concat(frames, ignore_index=True)
    uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()
    uni = uni.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)
    return uni

# ---------- Hybrid metadata enrichment (yahooquery → yfinance fallback) ----------

def _load_meta_cache(cache_path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame(columns=["ticker","name","industry","sector","market_cap"])

def _save_meta_cache(df: pd.DataFrame, cache_path: str):
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

def fetch_meta_yq_then_yf(
    tickers: List[str],
    cache_path: str = META_CACHE_PATH,
    batch: int = 150,
    max_workers: int = 8,
    retry: int = 0,
    backoff: float = 1.2,
    yf_sleep: float = 0,
    use_fallback: bool = True,
) -> pd.DataFrame:
    """
    1) Try yahooquery (vectorized) for name/sector/industry/market_cap.
    2) For tickers still missing sector/industry, optionally fallback to yfinance.
    3) Cache results to parquet for later runs.
    """
    from yahooquery import Ticker as YQ
    import yfinance as yf

    cache = _load_meta_cache(cache_path)
    cached = cache[cache["ticker"].isin(tickers)].copy()
    have = set(cached["ticker"])
    need = [t for t in tickers if t not in have]

    rows = []

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # --- yahooquery bulk ---
    if need:
        print(f"[META] yahooquery: total={len(tickers)} cached={len(cached)} need_fetch={len(need)}")
        for group in chunks(need, batch):
            for attempt in range(retry + 1):
                try:
                    tq = YQ(group, asynchronous=True, max_workers=max_workers, retry=2, backoff_factor=0.5)
                    price = tq.price or {}
                    prof  = tq.summary_profile or {}
                    for t in group:
                        p = price.get(t) or {}
                        pr = prof.get(t) or {}
                        rows.append({
                            "ticker": t,
                            "name": p.get("shortName") or p.get("longName") or t,
                            "industry": pr.get("industry"),
                            "sector": pr.get("sector"),
                            "market_cap": p.get("marketCap") or p.get("market_cap")
                        })
                    break  # success
                except Exception:
                    print("Exception here")
                    if attempt == retry:
                        for t in group:
                            rows.append({
                                "ticker": t, "name": t,
                                "industry": None, "sector": None, "market_cap": None
                            })
                    else:
                        time.sleep(backoff * (attempt + 1))

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cached.columns)
    meta = pd.concat([cached, new_df], ignore_index=True).drop_duplicates("ticker", keep="last")

    # --- yfinance fallback for missing sector/industry ---
    if use_fallback:
        missing = meta[meta["industry"].isna() & meta["sector"].isna()]
        if not missing.empty:
            print(f"[META] yfinance fallback for {len(missing)} tickers lacking sector/industry…")
        count = 1
        for t in missing["ticker"].tolist():
            try:
                tk = yf.Ticker(t)
                try:
                    info = tk.get_info()
                except Exception:
                    info = tk.info
                meta.loc[meta["ticker"] == t, ["name","industry","sector","market_cap"]] = [
                    info.get("shortName") or info.get("longName") or t,
                    info.get("industry"),
                    info.get("sector"),
                    info.get("marketCap") or info.get("enterpriseValue")
                ]
                time.sleep(yf_sleep)
                print(f'Fetched count: {count} / {len(missing)} || Now: {t} Industry: {info.get("industry")}')
                count += 1
            except Exception:
                print("Full Failure")
                # leave as None if still failing
                pass

    _save_meta_cache(meta, cache_path)
    print(meta)
    return meta[["ticker","name","industry","sector","market_cap"]]

# ---------- CSV writer: only update when changed ----------

def write_csv_if_changed(df: pd.DataFrame, path: str) -> bool:
    df_norm = df.copy()
    df_norm = df_norm[[c for c in sorted(df_norm.columns)]]
    new_hash = _sha256_df(df_norm)

    old_hash = None
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            old = old[[c for c in sorted(old.columns)]]
            old_hash = _sha256_df(old)
        except Exception:
            old_hash = None

    if new_hash == old_hash:
        print(f"No changes detected → leaving existing file untouched: {path}")
        return False

    df.to_csv(path, index=False)
    print(f"Saved → {path} (content changed)")
    return True

# ---------- Batch runner ----------

def run_batches(
    universe: pd.DataFrame,
    min_cap: Optional[float],
    max_cap: Optional[float],
    out_csv: str,
    days: int,
    recency_days: int,
    blurbs: bool,
    openai_key: Optional[str],
    gpt_model: str,
    batch_size: int,
    sleep_between_batches: float,
    meta_cache: str,
    keep_missing_cap: bool,
):
    all_tickers = universe["ticker"].tolist()
    print(f"Universe tickers: {len(all_tickers)}")
    print("Sample:", all_tickers[:10])

    if not all_tickers:
        print("[STOP] Empty universe. Writing empty CSV.")
        empty = pd.DataFrame(columns=["ticker","name","market_cap","activity_score","label"])
        write_csv_if_changed(empty, out_csv)
        return

    print("Fetching industry/cap via yahooquery → yfinance fallback (with cache)…")
    meta = fetch_meta_yq_then_yf(all_tickers, cache_path=meta_cache)

    if meta.empty:
        print("[STOP] Metadata fetch returned 0 rows. Writing empty CSV.")
        write_csv_if_changed(pd.DataFrame(columns=["ticker","name","market_cap","activity_score","label"]), out_csv)
        return

    def _is_bio_row(r: pd.Series) -> bool:
        return _is_biotech_text(r.get("industry",""), r.get("sector",""))

    bio_mask = meta.apply(_is_bio_row, axis=1)
    bio = meta[bio_mask].copy()

    if not keep_missing_cap:
        bio = bio[bio["market_cap"].notna()].copy()

    if min_cap is not None:
        bio = bio[bio["market_cap"].fillna(0) >= float(min_cap)]
    if max_cap is not None:
        bio = bio[bio["market_cap"].fillna(0) <= float(max_cap)]

    bio = bio.sort_values("market_cap", ascending=True).reset_index(drop=True)
    print(f"Biotech filtered: {len(bio)} tickers (cap ∈ [{min_cap}, {max_cap}], keep_missing_cap={keep_missing_cap})")

    if bio.empty:
        print("No biotech tickers in the selected cap band. Writing empty CSV.")
        write_csv_if_changed(pd.DataFrame(columns=["ticker","name","market_cap","activity_score","label"]), out_csv)
        return

    # Screen in manageable chunks
    chunks = chunked(bio["ticker"].tolist(), batch_size)
    out_frames = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] Screening {len(chunk)} tickers…")
        df = screen_activity_for_tickers(
            chunk,
            min_cap=min_cap,
            max_cap=max_cap,
            days=days,
            recency_days=recency_days,
            sleep_between=0.0,
            api_key=openai_key if blurbs else None,
            gpt_model=gpt_model,
            want_blurb=blurbs
        )
        out_frames.append(df)
        if sleep_between_batches and i < len(chunks):
            time.sleep(sleep_between_batches)

    out = pd.concat(out_frames, ignore_index=True)
    out = out.sort_values("activity_score", ascending=False)
    write_csv_if_changed(out, out_csv)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Screen US (+ optional ASX) biotechs with hybrid metadata fetch & idempotent CSV output.")
    ap.add_argument("--us", action="store_true", help="Include US (NASDAQ JSON universe)")
    ap.add_argument("--asx", action="store_true", help="Include ASX (requires a local CSV)")
    ap.add_argument("--asx-file", type=str, default=DEFAULT_ASX_LOCAL, help=f"Path to ASX CSV (default: {DEFAULT_ASX_LOCAL}, requires column 'code' or 'ticker')")
    ap.add_argument("--min-cap", type=float, default=0, help="Minimum market cap (e.g., 1.5e8)")
    ap.add_argument("--max-cap", type=float, default=2.5e9, help="Maximum market cap (e.g., 2.5e9)")
    ap.add_argument("--keep-missing-cap", action="store_true", help="Keep rows with missing market cap (otherwise dropped)")
    ap.add_argument("--days", type=int, default=365, help="News lookback window (days)")
    ap.add_argument("--recency-days", type=int, default=540, help="CT.gov recency window (days)")
    ap.add_argument("--out", type=str, default="biotech_activity_all.csv", help="Output CSV path")
    ap.add_argument("--blurbs", action="store_true", help="Generate OpenAI blurbs (requires --openai-key)")
    ap.add_argument("--openai-key", type=str, default=API_KEY, help="OpenAI API key (only needed if --blurbs)")
    ap.add_argument("--gpt-model", type=str, default="gpt-4o-mini", help="OpenAI model for blurbs")
    ap.add_argument("--batch-size", type=int, default=30, help="Tickers per screening batch")
    ap.add_argument("--sleep-batches", type=float, default=0.0, help="Sleep seconds between screening batches")
    ap.add_argument("--meta-cache", type=str, default=META_CACHE_PATH, help=f"Parquet path for metadata cache (default: {META_CACHE_PATH})")
    args = ap.parse_args()

    if not args.us and not args.asx:
        print("Select at least one exchange: use --us and/or --asx")
        sys.exit(1)
    if args.blurbs and not args.openai_key:
        print("You passed --blurbs but no --openai-key. Proceeding WITHOUT blurbs.")
        args.blurbs = False

    uni = build_universe(include_asx=args.asx, include_us=args.us, asx_file=args.asx_file)
    if uni.empty:
        print("[STOP] Built universe is empty. Check connectivity or provide local CSVs.")
        write_csv_if_changed(pd.DataFrame(columns=["ticker","name","market_cap","activity_score","label"]), args.out)
        sys.exit(0)

    run_batches(
        universe=uni,
        min_cap=args.min_cap,
        max_cap=args.max_cap,
        out_csv=args.out,
        days=args.days,
        recency_days=args.recency_days,
        blurbs=args.blurbs,
        openai_key=args.openai_key,
        gpt_model=args.gpt_model,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep_batches,
        meta_cache=args.meta_cache,
        keep_missing_cap=args.keep_missing_cap
    )

if __name__ == "__main__":
    main()
