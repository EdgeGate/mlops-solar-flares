# src/data/xrs_update.py
from __future__ import annotations
import argparse, os, re, sys, io, tempfile, time, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import logging
import json
import pandas as pd
import numpy as np
import requests
import xarray as xr


try:
    import sunpy.timeseries as ts
    from sunpy.net import Fido, attrs as a
    _SUNPY_OK = True
except Exception:
    _SUNPY_OK = False


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/project"))
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT = DATA_DIR / "xrs_clean.parquet"

# -------- utils ----------
# --- utils -------------------------------------------------
def _atomic_write_parquet(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)

    # NE supprimer que ces colonnes si elles sont 100% nulles
    df = _drop_fully_null_columns(df)

    # fichier temporaire dans le même dossier (évite "Invalid cross-device link")
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=".tmp-", suffix=".parquet")
    os.close(fd)
    tmp_path = Path(tmp)

    df.to_parquet(tmp_path, index=False)

    try:
        os.replace(tmp_path, out)
    except OSError:
        import shutil
        shutil.move(str(tmp_path), str(out))


# util: dropper uniquement certaines colonnes si 100% vides
def _drop_fully_null_columns(df: pd.DataFrame, cols=("source","quality_flag","satellite_longitude")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].isna().all():
            df = df.drop(columns=[c])
    return df


def _clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    want = ["time","flux_long_wm2","flux_short_wm2","satellite",
            "energy_long","energy_short","source","quality_flag","satellite_longitude",
            "date","hour","minute_of_day","dow"]
    for c in want:
        if c not in df.columns: df[c] = pd.NA
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last")
    for c in ["flux_long_wm2","flux_short_wm2","satellite_longitude"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["flux_long_wm2","flux_short_wm2"]:
        if c in df.columns:
            df = df[(df[c].isna()) | (df[c] >= 0)]
            if df[c].notna().any():
                hi = np.nanpercentile(df[c].astype(float), 99.9)
                if np.isfinite(hi): df[c] = df[c].clip(upper=hi).astype("float32")
    df["satellite"] = df["satellite"].astype("string")
    df["energy_long"] = "0.1-0.8 nm"
    df["energy_short"] = "0.05-0.4 nm"
    df["source"] = df["source"].astype("string")
    df["quality_flag"] = df["quality_flag"].astype("string")
    df["date"] = df["time"].dt.date.astype("string")
    df["hour"] = df["time"].dt.hour.astype("int16")
    df["minute_of_day"] = (df["time"].dt.hour*60 + df["time"].dt.minute).astype("int16")
    df["dow"] = df["time"].dt.weekday.astype("int8")
    return df[want]

# --- merge -------------------------------------------------
def _merge_to_out(batch: pd.DataFrame):
    batch = _clean_schema(batch)
    if OUT.exists():
        base = pd.read_parquet(OUT)
        if "xrs_1_8" in base.columns and "flux_long_wm2" not in base.columns:
            base = base.rename(columns={"xrs_1_8": "flux_long_wm2"})
            base["source"] = base.get("source", "unknown")
        base = _clean_schema(base)

        before_count = len(base)

        merged = pd.concat([base, batch], ignore_index=True).drop_duplicates(
            subset=["time"], keep="last"
        )
        merged = merged.dropna(axis=1, how='all')

        after_count = len(merged)
        added_count = after_count - before_count

        # 📌 log JSON
        logging.info(json.dumps({
            "merged_rows_total": after_count,
            "rows_added": added_count,
            "range": [
                str(merged["time"].min()),
                str(merged["time"].max())
            ]
        }))

        _atomic_write_parquet(merged, OUT)

    else:
        merged = batch
        logging.info(json.dumps({
            "merged_rows_total": len(merged),
            "rows_added": len(merged),
            "range": [
                str(merged["time"].min()),
                str(merged["time"].max())
            ]
        }))
        _atomic_write_parquet(merged, OUT)

    return merged["time"].min(), merged["time"].max(), len(merged)


# -------- SWPC (near real-time) ----------
SWPC_BASE = "https://services.swpc.noaa.gov/json/goes/primary"

def _norm_energy(e: str | None) -> str | None:
    if not isinstance(e, str):
        return None
    e = e.strip().lower().replace(" ", "")  # retire espaces éventuels
    # on remet le format canonique avec espace
    if e == "0.1-0.8nm":
        return "0.1-0.8 nm"
    if e in ("0.05-0.4nm", "0.5-4a", "0.5-4å"):  # au cas où…
        return "0.05-0.4 nm"
    return None  # on ignore les autres

def _norm_satellite(sat) -> str:
    # JSON SWPC a souvent 16/17/18 (int) ; parfois déjà 'GOES-18'
    if isinstance(sat, (int, float)) and int(sat) in range(1, 100):
        return f"GOES-{int(sat):02d}"
    if isinstance(sat, str):
        m = re.search(r"(\d{2})", sat)
        return f"GOES-{m.group(1)}" if m else sat.upper()
    return "unknown"

def _get_sat_longitudes_map() -> dict[str, float]:
    try:
        j = requests.get("https://services.swpc.noaa.gov/json/goes/satellite-longitudes.json", timeout=15).json()
        # ex: [{'satellite':'GOES-18','longitude':-137.2}, ...]
        return {str(row.get("satellite")).upper(): row.get("longitude") for row in j if isinstance(row, dict)}
    except Exception:
        return {}

def _fetch_swpc_json(minutes: int = 1500) -> pd.DataFrame:
    url = f"{SWPC_BASE}/xrays-7-day.json" if minutes > 24*60 else f"{SWPC_BASE}/xrays-3-day.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for obj in data if isinstance(data, list) else []:
        e = _norm_energy(obj.get("energy"))
        if e is None:
            continue
        sat = _norm_satellite(obj.get("satellite"))
        t = obj.get("time_tag")
        flux = obj.get("flux")
        rows.append({
            "time": t,
            "satellite": sat,
            "source": "SWPC",
            "flux_long_wm2": flux if e == "0.1-0.8 nm" else np.nan,
            "flux_short_wm2": flux if e == "0.05-0.4 nm" else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = (df.groupby(["time","satellite"], as_index=False)
            .agg({"flux_long_wm2":"max","flux_short_wm2":"max"}))

    sat_lon = _get_sat_longitudes_map()
    df["satellite_longitude"] = df["satellite"].map(lambda s: sat_lon.get(str(s).upper()))
    df["quality_flag"] = pd.NA
    return df

# -------- NCEI (historique massif) ----------
NCEI_BASE = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs"
SAT_LIST = ["goes14","goes15","goes16","goes17","goes18"]

def _list_nc_for_day(sat: str, day: datetime) -> list[str]:
    url = f"{NCEI_BASE}/{sat}/xrsf-l2-avg1m_science/{day:%Y}/{day:%m}/"
    try:
        html = requests.get(url, timeout=30).text
    except Exception:
        return []
    # tolérant aux -/_ et aux versions avec lettres/chiffres/points
    daystr = day.strftime("%Y%m%d")
    pat = re.compile(
        rf"xrsf[-_]l2[-_]avg1m[-_]science[_]g\d{{2}}[_]d{daystr}[_]v[\w.\-]+\.nc",
        re.I,
    )
    return [url + m.group(0) for m in pat.finditer(html)]

def _open_ncei_nc(url: str, sat_name: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60); r.raise_for_status()
    with xr.open_dataset(io.BytesIO(r.content)) as ds:
        vars_candidates = [v for v in ds.data_vars if "time" in ds[v].dims]
        long_var = next((v for v in vars_candidates if re.search(r"(xrsb|long|1_8|0\.1-0\.8)", v, re.I)), None)
        short_var = next((v for v in vars_candidates if re.search(r"(xrsa|short|0_5_0_4|0\.05-0\.4)", v, re.I)), None)
        df = ds.to_dataframe().reset_index()
    time_col = "time" if "time" in df.columns else df.columns[0]
    out = pd.DataFrame({"time": pd.to_datetime(df[time_col], utc=True, errors="coerce")})
    out["flux_long_wm2"] = pd.to_numeric(df.get(long_var), errors="coerce") if long_var in df else np.nan
    out["flux_short_wm2"] = pd.to_numeric(df.get(short_var), errors="coerce") if (short_var and short_var in df) else np.nan
    out["satellite"] = sat_name.upper().replace("GOES","GOES-")
    out["source"] = "NCEI"
    out["quality_flag"] = df.get("quality_flag")
    out["energy_long"] = "0.1-0.8 nm"
    out["energy_short"] = "0.05-0.4 nm"
    return out.dropna(subset=["time"]).sort_values("time")


def _open_sunpy_xrs_day(day: datetime) -> pd.DataFrame:
    if not _SUNPY_OK:
        return pd.DataFrame()

    from pathlib import Path
    start_day = pd.Timestamp(day.date(), tz="UTC")
    end_day   = start_day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    q_start   = start_day - pd.Timedelta(days=1)
    q_end     = end_day   + pd.Timedelta(days=1)

    # Requête
    try:
        res = Fido.search(a.Time(q_start.to_pydatetime(), q_end.to_pydatetime()),
                          a.Instrument('XRS'))
        if len(res) == 0:
            res = Fido.search(a.Time(q_start.to_pydatetime(), q_end.to_pydatetime()),
                              a.Instrument('XRS'), a.Provider('NOAA'))
    except Exception:
        return pd.DataFrame()
    if len(res) == 0:
        return pd.DataFrame()

    files = Fido.fetch(res, progress=False)

    # 1) Priorité aux fichiers 1-minute
    files_1m = [f for f in files if "avg1m" in Path(f).name.lower()]
    if files_1m:
        candidates = files_1m
        do_resample = False
    else:
        # 2) Fallback: 1-seconde → on rééchantillonne en 1-minute (moyenne)
        files_1s = [f for f in files if "flx1s" in Path(f).name.lower()]
        if not files_1s:
            return pd.DataFrame()
        candidates = files_1s
        do_resample = True

    frames = []
    for f in candidates:
        try:
            ts_obj = ts.TimeSeries(f)
        except Exception:
            continue
        df = ts_obj.to_dataframe().reset_index().rename(columns={"index": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        # colonnes XRS (xrsa = 0.05–0.4 nm ; xrsb = 0.1–0.8 nm)
        long_candidates  = [c for c in df.columns if re.search(r"(xrsb|long|1_8|0\.1|0_1)", c, re.I)]
        short_candidates = [c for c in df.columns if re.search(r"(xrsa|short|0\.05|0_05|0_5|0_4)", c, re.I)]
        long_col  = long_candidates[0]  if long_candidates  else None
        short_col = short_candidates[0] if short_candidates else None

        out = pd.DataFrame({
            "time": df["time"],
            "flux_long_wm2":  pd.to_numeric(df.get(long_col),  errors="coerce") if long_col  else pd.NA,
            "flux_short_wm2": pd.to_numeric(df.get(short_col), errors="coerce") if short_col else pd.NA,
        })

        if do_resample:
            # rééchantillonnage en moyenne 1-min (cohérent avec "avg1m")
            out = (out.set_index("time")
                       .resample("1min").mean()
                       .reset_index())

        frames.append(out)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Tronquer strictement au jour
    out = out[(out["time"] >= start_day) & (out["time"] <= end_day)]
    out["satellite"] = pd.NA
    out["source"] = "NCEI-SunPy"
    out["quality_flag"] = pd.NA
    out["energy_long"] = "0.1-0.8 nm"
    out["energy_short"] = "0.05-0.4 nm"
    return _clean_schema(out)


def backfill(start: datetime, end: datetime):
    cur = start; total = 0
    while cur <= end:
        day_frames = []
        urls_found = []
        for sat in SAT_LIST:
            urls_found.extend(_list_nc_for_day(sat, cur))

        # 1) .nc NCEI si trouvés
        if urls_found:
            for url in urls_found:
                try:
                    # déduit le sat si possible, sinon passe "unknown"
                    sat_guess = re.search(r"g(\d{2})", url, re.I)
                    sat_name = f"GOES-{sat_guess.group(1)}" if sat_guess else "unknown"
                    day_frames.append(_open_ncei_nc(url, sat_name))
                except Exception as e:
                    print(f"[WARN] NCEI open fail {url}: {e}")

        # 2) Fallback SunPy si rien obtenu
        if not day_frames:
            df_sp = _open_sunpy_xrs_day(cur)
            if not df_sp.empty:
                day_frames = [df_sp]

        if day_frames:
            df_day = pd.concat(day_frames, ignore_index=True)
            tmin, tmax, n = _merge_to_out(df_day); total = n
            print(json.dumps({"merged_rows_total": int(n),
                              "day": cur.strftime("%Y-%m-%d"),
                              "range":[str(tmin), str(tmax)]}, ensure_ascii=False))
        else:
            print(f"[WARN] Aucun fichier pour {cur:%Y-%m-%d} (NCEI listing vide et fallback SunPy sans résultat)")

        cur += timedelta(days=1)
    print(f"[DONE] backfill complete -> rows={total}")


def update_since(minutes: int, keep_days: Optional[int]):
    df = _fetch_swpc_json(minutes=minutes)
    if df.empty:
        print("[INFO] SWPC returned empty"); return
    tmin, tmax, n = _merge_to_out(df)
    if keep_days:
        base = pd.read_parquet(OUT)
        now_utc = pd.Timestamp.utcnow()
        now_utc = now_utc if now_utc.tz is not None else now_utc.tz_localize("UTC")
        cutoff = now_utc - pd.Timedelta(days=keep_days)

        base = base[base["time"] >= cutoff]
        _atomic_write_parquet(base, OUT)
        print(f"[INFO] retention applied: >= {cutoff}")
    print(json.dumps({"merged_rows_total": int(n), "range":[str(tmin), str(tmax)]}, ensure_ascii=False))

# -------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backfill","update"], required=True)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--since-minutes", type=int, default=1500)
    ap.add_argument("--keep-days", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "backfill":
        if not (args.start and args.end): ap.error("--start and --end are required for backfill")
        start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        backfill(start, end)
    else:
        update_since(args.since_minutes, args.keep_days)

if __name__ == "__main__":
    main()
