import os
from time import perf_counter

import pandas as pd
import streamlit as st

from src.crew_pipeline import PipelineResult, answer_sales_question, get_warehouse


st.set_page_config(page_title="Sales AI Copilot", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .main-title {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.2rem;}
      .subtitle {color: #4b5563; margin-bottom: 1.0rem;}
      .stChatMessage {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Sales AI Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Schema-aware analytics with validation agents for SQL quality, result quality, and visualization correctness.</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_latency" not in st.session_state:
    st.session_state.last_latency = 0.0


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False), errors="coerce")


def render_visualization(result: dict, df: pd.DataFrame) -> None:
    viz = result.get("visualization", {})
    if not viz or not viz.get("required"):
        return

    chart_type = viz.get("chart_type", "none")
    x_col = viz.get("x")
    y_cols = viz.get("y", [])

    if not x_col or x_col not in df.columns or not y_cols:
        st.info("Visualization skipped due to invalid chart mapping.")
        return

    valid_y = [c for c in y_cols if c in df.columns]
    if not valid_y:
        st.info("Visualization skipped due to missing metric columns.")
        return

    chart_df = df[[x_col] + valid_y].copy()
    for col in valid_y:
        chart_df[col] = _to_numeric(chart_df[col])
    chart_df = chart_df.dropna(subset=valid_y)

    if chart_df.empty:
        st.info("Visualization skipped because chart data is empty.")
        return

    st.subheader("Visualization")
    chart_df = chart_df.set_index(x_col)
    if chart_type == "bar":
        st.bar_chart(chart_df)
    else:
        st.line_chart(chart_df)

    notes = viz.get("notes", [])
    st.caption(f"Viz validation status: {viz.get('status', 'unknown')}")
    for n in notes[:3]:
        st.caption(f"- {n}")


def render_result(result: dict) -> None:
    st.subheader("SQL Used")
    st.code(result.get("sql", ""), language="sql")

    st.subheader("Result Table")
    rows = result.get("table_records", [])
    cols = result.get("table_columns", [])
    if rows and cols:
        df = pd.DataFrame(rows, columns=cols)
        st.dataframe(df, use_container_width=True, hide_index=True)
        render_visualization(result, df)
    else:
        st.info("No rows returned.")

    st.subheader("Validation")
    st.markdown(f"- Status: `{result.get('validation_status', 'unknown')}`")
    for note in result.get("validation_notes", []):
        st.markdown(f"- {note}")

    st.subheader("Insights")
    for bullet in result.get("insights", []):
        st.markdown(f"- {bullet}")

    next_q = result.get("next_question", "")
    if next_q:
        st.subheader("Next Question")
        st.markdown(next_q)


with st.sidebar:
    st.subheader("Runtime")
    mode_label = st.radio("Pipeline mode", ["Fast (Low Latency)", "Deep (Higher Quality)"], index=0)
    mode = "fast" if mode_label.startswith("Fast") else "deep"

    st.subheader("Model")
    model_name = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    os.environ["OPENAI_MODEL"] = model_name.strip() or "gpt-4o-mini"

    st.subheader("Quick Prompts")
    quick_prompts = [
        "Top 10 states by Amazon sales amount and total units",
        "Show cancellation rate by fulfillment channel",
        "Give a visualization of total sales trend",
        "Give 5 lines of insights for data of each file",
    ]
    for qp in quick_prompts:
        if st.button(qp):
            st.session_state["queued_prompt"] = qp

    st.subheader("System")
    if st.button("Show Schema + Dataset Playbook"):
        wh = get_warehouse()
        st.text(wh.schema_text())
        st.text(wh.dataset_playbook())

cols = st.columns(2)
cols[0].metric("Pipeline", mode)
cols[1].metric("Last Latency (s)", f"{st.session_state.last_latency:.2f}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("type") == "result":
            render_result(message["payload"])
        else:
            st.markdown(message["content"])

queued_prompt = st.session_state.pop("queued_prompt", "") if "queued_prompt" in st.session_state else ""
prompt = st.chat_input("Ask: state rankings, cancellation rates, AOV, trend visuals, dataset insights...")
if not prompt and queued_prompt:
    prompt = queued_prompt

if prompt:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("`OPENAI_API_KEY` is not set.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        ui_start = perf_counter()
        with st.spinner("Running validated analysis..."):
            result: PipelineResult = answer_sales_question(prompt, mode=mode)
        ui_elapsed = perf_counter() - ui_start
        st.session_state.last_latency = result.elapsed_seconds

        payload = {
            "sql": result.sql,
            "table_records": result.table_records,
            "table_columns": result.table_columns,
            "insights": result.insights,
            "next_question": result.next_question,
            "source": result.source,
            "validation_status": result.validation_status,
            "validation_notes": result.validation_notes,
            "visualization": result.visualization,
        }
        render_result(payload)

        with st.expander("Diagnostics"):
            st.write(
                {
                    "pipeline_mode": result.mode,
                    "source": result.source,
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "pipeline_elapsed_seconds": round(result.elapsed_seconds, 3),
                    "ui_elapsed_seconds": round(ui_elapsed, 3),
                }
            )

    st.session_state.messages.append({"role": "assistant", "type": "result", "payload": payload})
