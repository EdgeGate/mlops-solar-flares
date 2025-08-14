# airflow/dags/xrs_update_daily.py
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

ROOT = os.getenv("PROJECT_ROOT", "/opt/project")
ENV  = {"PROJECT_ROOT": "/opt/project"}

update_swpc_recent = BashOperator(
    task_id="update_swpc_recent",
    bash_command=f"python {ROOT}/src/data/xrs_update.py --mode update --since-minutes 20160 --keep-days 180",
    env=ENV,
)



default_args = {
    "owner": "you",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="xrs_update_12h",
    start_date=datetime(2025, 1, 1),
    schedule="0 */12 * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["xrs", "noaa", "update"],
) as dag:

    backfill_j2 = BashOperator(
        task_id="backfill_ncei_j2",
        bash_command=(
            f"python {ROOT}/src/data/xrs_update.py "
            "--mode backfill "
            "--start {{ macros.ds_add(ds, -2) }} "
            "--end {{ macros.ds_add(ds, -2) }}"
        ),
        env=ENV,
        execution_timeout=timedelta(minutes=30),
        depends_on_past=False,
    )

    update_recent = BashOperator(
        task_id="update_swpc_recent",
        bash_command=(
            f"python {ROOT}/src/data/xrs_update.py "
            "--mode update "
            "--since-minutes 20160 "
            "--keep-days 180"
        ),
        env=ENV,
        execution_timeout=timedelta(minutes=15),
        depends_on_past=False,
    )

    # Guard: do not fail if the file is not there (first run)
    dq_quick = BashOperator(
        task_id="dq_quick",
        bash_command=(
            "python - <<'PY'\n"
            "import os, sys, pandas as pd\n"
            "root=os.environ.get('PROJECT_ROOT','/opt/project')\n"
            "p=os.path.join(root,'data','xrs_clean.parquet')\n"
            "print('reading:', p)\n"
            "if not os.path.exists(p):\n"
            "    print('file not found yet — skipping DQ'); sys.exit(0)\n"
            "df=pd.read_parquet(p)\n"
            "print('shape:', df.shape)\n"
            "print('nulls:\\n', df.isna().sum().sort_values(ascending=False).to_string())\n"
            "print('by satellite:\\n', df.groupby('satellite')['time'].agg(['min','max','count']))\n"
            "PY"
        ),
        env=ENV,
        depends_on_past=False,
    )

    backfill_j2 >> update_recent >> dq_quick
