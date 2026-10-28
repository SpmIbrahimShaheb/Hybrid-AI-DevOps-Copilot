import os
"""
dashboard/app.py
-----------------
Streamlit dashboard showing the full pipeline journey:
Raw Logs -> Clustered Incidents -> AI Diagnosis
"""

import sqlite3
import pandas as pd
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs.db")

st.set_page_config(
    page_title="Hybrid AI DevOps Copilot",
    page_icon="🛠️",
    layout="wide",
)


@st.cache_data(ttl=30)
def load_data(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    logs_df = pd.read_sql_query(
        "SELECT timestamp, service_name, log_level, message, cluster_id FROM logs ORDER BY timestamp",
        conn,
    )
    try:
        diagnoses_df = pd.read_sql_query(
            "SELECT cluster_id, diagnosis, log_count, source FROM diagnoses ORDER BY cluster_id",
            conn,
        )
    except Exception:
        diagnoses_df = pd.DataFrame(columns=["cluster_id", "diagnosis", "log_count", "source"])
    conn.close()
    return logs_df, diagnoses_df


def compute_metrics(logs_df):
    total_logs = len(logs_df)
    real_incidents = logs_df[logs_df["cluster_id"] != -1]["cluster_id"].nunique() \
        if "cluster_id" in logs_df.columns else 0
    noise_points = len(logs_df[logs_df["cluster_id"] == -1]) if "cluster_id" in logs_df.columns else 0
    reduction_pct = (1 - real_incidents / total_logs) * 100 if total_logs else 0
    return total_logs, real_incidents, noise_points, reduction_pct


def main():
    st.title("🛠️ Hybrid AI DevOps Copilot")
    st.caption("ML-based log clustering + LLM-assisted root cause diagnosis — built on 100% synthetic data, zero-cost tools.")

    try:
        logs_df, diagnoses_df = load_data()
    except Exception as e:
        st.error(f"Could not load {DB_PATH}. Make sure it's in the same folder as this app, "
                  f"and that generate_logs.py -> cluster_engine.py -> diagnose.py have all been run. "
                  f"Error: {e}")
        return

    if logs_df.empty:
        st.warning("No log data found. Run generate_logs.py first.")
        return

    total_logs, real_incidents, noise_points, reduction_pct = compute_metrics(logs_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Raw Log Lines", total_logs)
    col2.metric("Distinct Incidents Found", real_incidents)
    col3.metric("Noise/Unrelated Points", noise_points)
    col4.metric("Noise Reduction", f"{reduction_pct:.0f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Raw Logs", "🔗 Clustered Incidents", "🧠 AI Diagnosis"])

    with tab1:
        st.subheader("Raw Log Stream")
        st.caption("Every individual log line, before any grouping -- this is what an on-call engineer would otherwise have to review one by one.")
        level_filter = st.multiselect(
            "Filter by log level", options=sorted(logs_df["log_level"].unique()),
            default=list(logs_df["log_level"].unique()),
        )
        filtered = logs_df[logs_df["log_level"].isin(level_filter)]
        st.dataframe(filtered, use_container_width=True, height=500)

    with tab2:
        st.subheader("Clustered Incidents")
        st.caption("Related log lines grouped into single incidents by the ML clustering engine. Cluster -1 = unclustered noise.")
        if "cluster_id" not in logs_df.columns:
            st.warning("No cluster_id column found -- run cluster_engine.py first.")
        else:
            cluster_summary = (
                logs_df.groupby("cluster_id")
                .agg(log_count=("message", "count"), services=("service_name", lambda x: ", ".join(sorted(set(x)))))
                .reset_index()
                .sort_values("cluster_id")
            )
            cluster_summary["label"] = cluster_summary["cluster_id"].apply(
                lambda x: "Noise/Unclustered" if x == -1 else f"Incident #{x}"
            )
            st.dataframe(
                cluster_summary[["label", "log_count", "services"]],
                use_container_width=True,
            )

            selected_cluster = st.selectbox(
                "Inspect a cluster's raw log lines",
                options=cluster_summary["cluster_id"].tolist(),
                format_func=lambda x: "Noise/Unclustered" if x == -1 else f"Incident #{x}",
            )
            st.dataframe(
                logs_df[logs_df["cluster_id"] == selected_cluster][
                    ["timestamp", "service_name", "log_level", "message"]
                ],
                use_container_width=True,
            )

    with tab3:
        st.subheader("AI-Generated Root Cause Diagnosis")
        st.caption("Each incident's log lines, explained in plain English by the LLM layer, with a recommended first action.")
        if diagnoses_df.empty:
            st.warning("No diagnoses found -- run diagnose.py first.")
        else:
            for _, row in diagnoses_df.sort_values("cluster_id").iterrows():
                label = "Noise/Unclustered" if row["cluster_id"] == -1 else f"Incident #{row['cluster_id']}"
                with st.expander(f"{label}  —  {row['log_count']} log lines  (source: {row['source']})"):
                    st.write(row["diagnosis"])


if __name__ == "__main__":
    main()
