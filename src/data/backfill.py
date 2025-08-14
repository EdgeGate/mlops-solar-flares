# /opt/project/src/data/backfill.py
import os
import sys
import subprocess
import pandas as pd
PYTHON = os.getenv("BF_PY", "python")

PARQUET = "/opt/project/data/xrs_clean.parquet"

# Fenêtre cible (UTC) – surcharge possible via variables d'env BF_START / BF_END
WANT_START = pd.Timestamp(os.getenv("BF_START", "2025-07-01"), tz="UTC").normalize()
WANT_END   = pd.Timestamp(os.getenv("BF_END",   "2025-08-09"), tz="UTC").normalize()

# Sens d'extension CONTINUE
DIRECTION = os.environ.get("BF_DIR", "before").lower()
if DIRECTION not in {"before", "after"}:
    print("DIRECTION inconnue:", DIRECTION)
    sys.exit(2)

# Limiter le nombre de journées traitées en une exécution (sécurité)
BF_MAX_DAYS = int(os.getenv("BF_MAX_DAYS", "90"))

def _read_time_col(path: str) -> pd.Series:
    if not os.path.exists(path):
        return pd.Series(dtype="datetime64[ns, UTC]")
    t = pd.read_parquet(path, columns=["time"])["time"]
    t = pd.to_datetime(t, utc=True, errors="coerce").dropna()
    return t

def _days_present_set(t: pd.Series) -> set:
    if t.empty:
        return set()
    return set(t.dt.date)

def _call_backfill_day(day: pd.Timestamp) -> int:
    """Retourne le code de sortie du sous-processus backfill pour une journée."""
    d = day.date().isoformat()
    print(f"[CALL] backfill {d}")
    return subprocess.call([
        PYTHON, "/opt/project/src/data/xrs_update.py",
        "--mode", "backfill",
        "--start", d,
        "--end", d,
    ])

def _stop_with(msg: str, code: int = 0):
    print(msg)
    sys.exit(code)

def main():
    t = _read_time_col(PARQUET)
    days_present = _days_present_set(t)
    cur_min = t.min() if not t.empty else None
    cur_max = t.max() if not t.empty else None

    # Cas fichier inexistant : on part du bord demandé et on avance jour par jour
    if cur_min is None or cur_max is None:
        print("[INIT] xrs_clean.parquet absent ou vide — remplissage continu depuis la fenêtre demandée")
        # Choisir un sens par défaut cohérent (before => on remonte depuis WANT_END; after => on part de WANT_START)
        if DIRECTION == "before":
            cursor = WANT_END
            target_limit = WANT_START
            step = -1
        else:  # after
            cursor = WANT_START
            target_limit = WANT_END
            step = +1
        days_done = 0
        while days_done < BF_MAX_DAYS:
            if (step == -1 and cursor < target_limit) or (step == +1 and cursor > target_limit):
                break
            # Si la journée est déjà présente, on décale (continuité déjà satisfaite pour ce jour)
            if cursor.date() in days_present:
                cursor = cursor + pd.Timedelta(days=step)
                continue
            ret = _call_backfill_day(cursor)
            if ret != 0:
                _stop_with(f"[STOP] backfill {cursor.date()} a échoué (code={ret}) — on s'arrête pour éviter les trous", ret)
            # Rechargement pour vérifier que la journée est bien arrivée
            t2 = _read_time_col(PARQUET)
            new_days = _days_present_set(t2)
            if pd.to_datetime(cursor) not in new_days:
                _stop_with(f"[STOP] la journée {cursor.date()} n'est pas apparue après backfill — on s'arrête (pas de trou).", 1)
            days_present = new_days
            days_done += 1
            cursor = cursor + pd.Timedelta(days=step)

        # Bilan
        t_final = _read_time_col(PARQUET)
        _stop_with(f"Nouveau range : {t_final.min()} -> {t_final.max()} | rows = {len(t_final)}")

    # Fichier existant : extension CONTINUE stricte
    if DIRECTION == "before":
        # On veut remplir jour par jour avant cur_min, sans sauter de journée
        cursor = (cur_min - pd.Timedelta(days=1)).normalize()
        limit  = WANT_START
        step   = -1
        direction_label = "vers le passé"
    else:
        cursor = (cur_max + pd.Timedelta(days=1)).normalize()
        limit  = WANT_END
        step   = +1
        direction_label = "vers le futur"

    print(f"Plan {direction_label} : cible {limit.date()} — départ {cursor.date()} | existant: [{cur_min} → {cur_max}]")

    days_done = 0
    while days_done < BF_MAX_DAYS:
        # borne atteinte ?
        if (step == -1 and cursor < limit) or (step == +1 and cursor > limit):
            break
        # si la prochaine journée contiguë est déjà là, on est arrivé à la continuité demandée
        if cursor.date() in days_present:
            print(f"[OK] Continuité atteinte (jour {cursor.date()} déjà présent).")
            break

        # backfill d'une seule journée
        ret = _call_backfill_day(cursor)
        if ret != 0:
            _stop_with(f"[STOP] backfill {cursor.date()} a échoué (code={ret}) — on s'arrête pour éviter les trous", ret)

        # vérifier que la journée est bien arrivée
        t2 = _read_time_col(PARQUET)
        new_days = _days_present_set(t2)
        if pd.to_datetime(cursor) not in new_days:
            _stop_with(f"[STOP] la journée {cursor.date()} n'est pas apparue après backfill — on s'arrête (pas de trou).", 1)

        # ok, on continue à la journée adjacente suivante
        days_present = new_days
        days_done += 1
        cursor = cursor + pd.Timedelta(days=step)

    # Bilan final
    t_final = _read_time_col(PARQUET)
    print("Nouveau range :", t_final.min(), "->", t_final.max(), "| rows =", len(t_final))

if __name__ == "__main__":
    main()
