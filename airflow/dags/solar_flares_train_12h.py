# airflow/dags/solar_flares_train_12h.py
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

ROOT = os.getenv("PROJECT_ROOT", "/opt/project")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

ENV = {
    "PROJECT_ROOT": ROOT,
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    "MPLBACKEND": "Agg",
    "PYTHONUNBUFFERED": "1",
}

default_args = {"owner": "you", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="solar_flares_train_12h",
    start_date=datetime(2025, 1, 1),
    schedule_interval="10 0,12 * * *",   
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["train", "mlflow", "papermill"],
) as dag:

    # 0) ping: assure que l’expérience existe
    mlflow_ping = BashOperator(
        task_id="mlflow_ping",
        bash_command="""
python - <<'PY'
import os, mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
name="solar-flares"
client = MlflowClient()
exp = client.get_experiment_by_name(name)
if exp is None:
    client.create_experiment(name)
print("OK: experiment exists")
PY
""",
        env=ENV,
    )

    # 1) kernel pour Papermill
    ensure_kernel = BashOperator(
        task_id="ensure_kernel",
        bash_command="""
set -e
HOME=/home/airflow
python -c "import ipykernel" 2>/dev/null || pip install --no-cache-dir ipykernel==6.29.5
python -m ipykernel install --user --name airflow --display-name 'Python 3 (Airflow)'
KDIR="$HOME/.local/share/jupyter/kernels/airflow"
mkdir -p "$KDIR"
PYBIN="$(command -v python)"
cat > "$KDIR/kernel.json" <<JSON
{"argv":["$PYBIN","-Xfrozen_modules=off","-m","ipykernel_launcher","-f","{connection_file}"],
 "display_name":"Python 3 (Airflow)","language":"python"}
JSON
echo "Kernel airflow ready: $KDIR"
""",
        env={**ENV, "HOME": "/home/airflow"},
    )

    # 2) exécution notebook (fichier de sortie écrasé à chaque run)
    run_notebook = BashOperator(
        task_id="run_notebook",
        bash_command=f"""
set -euo pipefail
cd {ROOT}/notebooks
OUT="ml_x_ray_sensor_generated.ipynb"
TMP="$(mktemp -p /tmp ml_x_tmp_XXXXXX.ipynb)"
python -m papermill -k airflow ml_x_ray_sensor.ipynb "$TMP"
mv -f "$TMP" "$OUT"
echo "Notebook généré -> $OUT"
""",
        env=ENV,
        execution_timeout=timedelta(hours=2),
    )

    # 3) vérification stricte MLflow
    def verify_mlflow(mlflow_uri: str, experiment_name: str, model_name: str):
        import os, time, mlflow
        from mlflow import MlflowClient

        # si tu veux vraiment passer par env:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)

        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        assert exp, f"Expérience '{experiment_name}' introuvable"
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
        assert runs, "Aucun run MLflow trouvé"
        run = runs[0]; rid = run.info.run_id

        for m in ("test_f1_macro", "test_acc"):
            assert m in run.data.metrics, f"Métrique manquante: {m}"

        try:
            names = {a.path for a in client.list_artifacts(rid, path="evidently")}
        except Exception:
            names = set()
        assert ("evidently/evidently_report.html" in names) or ("evidently/evidently_report.json" in names), \
               "Rapport Evidently manquant"

        vers = client.search_model_versions(f"name='{model_name}'")
        assert vers, f"Aucune version dans le Model Registry '{model_name}'"
        latest = max(vers, key=lambda v: int(v.version))
        for _ in range(30):
            mv = client.get_model_version(model_name, latest.version)
            if mv.status == "READY":
                break
            time.sleep(2)
        mv = client.get_model_version(model_name, latest.version)
        assert mv.status == "READY", f"Model version v{latest.version} pas READY (status={mv.status})"
        print(f"✅ Vérif MLflow OK: run {rid}, version v{latest.version} READY")

    verify_mlflow_task = PythonOperator(
        task_id="verify_mlflow",
        python_callable=verify_mlflow,
        op_kwargs={
            "mlflow_uri": MLFLOW_URI,
            "experiment_name": "solar-flares",
            "model_name": "solar-flares-classifier",
        },
    )

    mlflow_ping >> ensure_kernel >> run_notebook >> verify_mlflow_task
