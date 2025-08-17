#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone

CLASSES = ["A", "B", "C", "M", "X"]

def to_utc_series_any(s: pd.Series) -> pd.Series:
    """Convertit une colonne temporelle en tz-aware UTC (supporte naive/aware)."""
    s = pd.to_datetime(s, errors="coerce", utc=False)
    # si naive => localize UTC; si aware => convert en UTC
    if getattr(getattr(s, "dt", None), "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")

def pick_time_col(df: pd.DataFrame) -> str:
    for c in ("time", "timestamp", "when_utc", "date"):
        if c in df.columns:
            return c
    raise KeyError("Colonne temps introuvable (attendu: time/timestamp/when_utc/date).")

def classify_from_flux(flux):
    if pd.isna(flux): return None
    if flux < 1e-7:   return "A"
    if flux < 1e-6:   return "B"
    if flux < 1e-5:   return "C"
    if flux < 1e-4:   return "M"
    return "X"

def ensure_label_col(df: pd.DataFrame) -> str:
    for c in ("flare_class","target","y_true","obs","class"):
        if c in df.columns:
            return c
    if "flux_long_wm2" not in df.columns:
        raise KeyError("Aucune colonne de label et 'flux_long_wm2' absent — impossible de déduire la classe.")
    df["flare_class"] = df["flux_long_wm2"].apply(classify_from_flux)
    return "flare_class"

def pct_distribution(series: pd.Series):
    if len(series) == 0:
        return {k: 0.0 for k in CLASSES}
    p = (series.astype(str).value_counts(normalize=True) * 100.0)\
            .reindex(CLASSES, fill_value=0.0)\
            .round(2)
    return {k: float(p[k]) for k in CLASSES}

def main():
    ap = argparse.ArgumentParser(
        description="Affiche le pourcentage de classes (A,B,C,M,X) observées dans les N dernières heures."
    )
    ap.add_argument("--file", default="data/xrs_clean.parquet", help="Chemin du parquet (défaut: data/xrs_clean.parquet)")
    ap.add_argument("--hours", type=int, default=12, help="Fenêtre glissante en heures (défaut: 12)")
    args = ap.parse_args()

    # Charge les observations
    df = pd.read_parquet(args.file)

    # Colonne temps → UTC tz-aware
    tcol = pick_time_col(df)
    df[tcol] = to_utc_series_any(df[tcol])

    # Fenêtre: maintenant (UTC) - hours → maintenant (UTC)
    now_utc = datetime.now(timezone.utc)
    t0 = now_utc - timedelta(hours=args.hours)
    win = df[(df[tcol] >= t0) & (df[tcol] <= now_utc)].copy()

    # Colonne label (reconstruite si besoin)
    ycol = ensure_label_col(win)

    # Distribution %
    dist = pct_distribution(win[ycol])

    # Affichage propre
    print(f"📅 Fenêtre UTC: {t0} → {now_utc}  (durée ~{args.hours}h)")
    print(f"📦 Fichier    : {args.file}")
    print(f"🔢 N mesures  : {len(win)}")
    print("— Répartition sur la fenêtre : " + "  ".join([f"{c} {dist[c]}%" for c in CLASSES]))

if __name__ == "__main__":
    main()


# Utile pour comparer les résultats,
# à condition d’être exécuté dans la fenêtre de 5 minutes
# qui suit l’exécution du DAG xrs_clean
# et précède celle du DAG ml_x_ray_sensor.
