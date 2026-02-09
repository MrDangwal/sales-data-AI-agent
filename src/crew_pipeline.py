from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from src.sales_data import SalesWarehouse, TableProfile, create_warehouse


WAREHOUSE: SalesWarehouse | None = None
NORMALIZE_SPACE_PATTERN = re.compile(r"\s+")
SQL_BLOCK_PATTERN = re.compile(r"```sql\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
TOP_N_PATTERN = re.compile(r"\btop\s+(\d+)\b", flags=re.IGNORECASE)

# Keep CrewAI storage local and writable for stable agent execution.
if "CREWAI_STORAGE_DIR" not in os.environ:
    os.environ["CREWAI_STORAGE_DIR"] = str(Path(".cache/crewai_storage").resolve())
Path(os.environ["CREWAI_STORAGE_DIR"]).mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineResult:
    sql: str
    table_records: list[dict]
    table_columns: list[str]
    insights: list[str]
    next_question: str
    elapsed_seconds: float
    mode: str
    source: str
    validation_status: str
    validation_notes: list[str]
    visualization: dict[str, Any]


def _qi(ident: str) -> str:
    return "\"" + ident.replace("\"", "\"\"") + "\""


def _build_llm(model_name: str):
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45")),
            max_retries=1,
        )
    except Exception:
        return model_name


def get_warehouse() -> SalesWarehouse:
    global WAREHOUSE
    if WAREHOUSE is None:
        WAREHOUSE = create_warehouse("Sales Dataset")
    return WAREHOUSE


@tool("Get relevant sales schema")
def get_relevant_sales_schema(question: str) -> str:
    """Return table-level semantic catalog for top relevant tables based on the question."""
    return get_warehouse().semantic_catalog_text(question, top_n=4)


@tool("Get dataset playbook")
def get_dataset_playbook(_: str = "") -> str:
    """Return adaptive dataset playbook and table routing principles."""
    return get_warehouse().dataset_playbook()


def _normalize_question(question: str) -> str:
    return NORMALIZE_SPACE_PATTERN.sub(" ", question.strip().lower())


def _extract_sql(text: str) -> str:
    block = SQL_BLOCK_PATTERN.search(text)
    if block:
        return block.group(1).strip().rstrip(";")

    pos = text.lower().find("select")
    if pos >= 0:
        return text[pos:].strip().rstrip(";")

    pos = text.lower().find("with")
    if pos >= 0:
        return text[pos:].strip().rstrip(";")

    return ""


def _table_to_payload(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    formatted = get_warehouse().format_dataframe(df)
    return formatted.to_dict(orient="records"), list(formatted.columns)


def _basic_insights(df: pd.DataFrame, question: str) -> list[str]:
    if df.empty:
        return ["No rows returned for this query. Validator attempted corrective fallback."]

    insights: list[str] = [f"Returned {len(df)} rows across {len(df.columns)} columns."]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        primary = None
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                primary = col
                break

        if primary:
            series = pd.to_numeric(df[primary], errors="coerce")
            insights.append(f"`{primary}` range: {series.min():,.2f} to {series.max():,.2f}.")
            if non_numeric_cols:
                idx = series.idxmax()
                label_col = non_numeric_cols[0]
                insights.append(f"Top `{label_col}` by `{primary}`: `{df.loc[idx, label_col]}`.")

    q = question.lower()
    if "cancel" in q:
        cancel_cols = [c for c in df.columns if "cancel" in c.lower() or "rate" in c.lower()]
        if cancel_cols:
            insights.append(f"Cancellation metric surfaced via `{cancel_cols[0]}`.")
    if any(k in q for k in ["trend", "month", "date"]):
        time_cols = [c for c in df.columns if any(t in c.lower() for t in ["date", "month", "period", "time"])]
        if time_cols:
            insights.append(f"Trend axis identified as `{time_cols[0]}`.")

    return insights[:5]


def _suggest_next_question(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["trend", "month", "date"]):
        return "Would you like the same trend segmented by top categories?"
    if "state" in q or "city" in q:
        return "Do you want a month-wise trend for these top geographies?"
    if "category" in q:
        return "Should I add cancellation rate and volume beside the category metric?"
    if "expense" in q:
        return "Do you want a cumulative cashflow trend with running balance?"
    if "mrp" in q or "tp" in q or "price" in q:
        return "Should I calculate spread percent and outlier products for this pricing view?"
    return "Do you want this analysis compared against another relevant dataset?"


def _limit_from_question(question: str, default: int = 10) -> int:
    m = TOP_N_PATTERN.search(question)
    if m:
        return max(1, min(int(m.group(1)), 100))
    return default


def _contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def _find_col(columns: list[str], contains: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    lowered = [(c, c.lower()) for c in columns]

    for token in contains:
        for col, low in lowered:
            if token in low and all(x not in low for x in exclude):
                return col

    return None


def _pick_time_col(profile: TableProfile) -> str | None:
    if profile.date_cols:
        best = _find_col(profile.date_cols, ["month", "period", "date", "time", "year"])
        return best or profile.date_cols[0]

    return _find_col(profile.columns, ["month", "period", "date", "time", "year"]) or _find_col(
        profile.columns, ["_date", "_month"]
    )


def _pick_dimension_col(profile: TableProfile, question: str) -> str | None:
    q = question.lower()
    preferred_tokens = [
        ("particular", ["expense_particular", "particular"]),
        ("state", ["state", "region", "province"]),
        ("city", ["city"]),
        ("category", ["category", "segment", "type"]),
        ("customer", ["customer", "client", "buyer"]),
        ("style", ["style", "design"]),
        ("sku", ["sku", "product", "item"]),
        ("fulfillment", ["fulfillment", "channel", "provider"]),
        ("size", ["size"]),
        ("color", ["color"]),
    ]

    for trigger, tokens in preferred_tokens:
        if trigger in q:
            col = _find_col(profile.columns, tokens, exclude=["_num", "_date"])
            if col:
                return col

    if "trend" in q or "month" in q or "date" in q:
        t = _pick_time_col(profile)
        if t:
            return t

    fallback = [
        c
        for c in profile.text_cols
        if not c.lower().endswith(("_num", "_date")) and c.lower() not in {"index", "id"}
    ]
    if fallback:
        return fallback[0]

    if profile.columns:
        return profile.columns[0]

    return None


def _pick_measure_col(profile: TableProfile, question: str) -> str | None:
    q = question.lower()
    numeric_cols = [c for c in profile.numeric_cols if c.lower() not in {"index", "id"} and not c.lower().endswith("_id")]
    if not numeric_cols:
        numeric_cols = profile.numeric_cols
    if not numeric_cols:
        return None

    preference_groups = [
        (["expense", "cost", "spend"], ["expense", "cost", "spend"]),
        (["received", "income", "inflow"], ["received", "income", "inflow"]),
        (["sales", "revenue", "amount", "gross", "value", "income"], ["sales", "revenue", "amount", "gross", "value"]),
        (["qty", "quantity", "units", "volume", "pcs", "stock"], ["qty", "quantity", "units", "volume", "pcs", "stock"]),
        (["rate", "price", "mrp", "tp", "cost", "margin"], ["rate", "price", "mrp", "tp", "cost", "margin"]),
    ]

    for triggers, col_tokens in preference_groups:
        if _contains_any(q, triggers):
            col = _find_col(numeric_cols, col_tokens)
            if col:
                return col

    return numeric_cols[0]


def _pick_secondary_measure_col(profile: TableProfile, primary: str | None) -> str | None:
    if not primary:
        return None
    for col in profile.numeric_cols:
        if col != primary:
            return col
    return None


def _build_cross_table_trend_sql(question: str, ranked_tables: list[str]) -> str | None:
    wh = get_warehouse()
    profiles = wh.table_profiles()
    if len(ranked_tables) < 2:
        return None

    t1, t2 = ranked_tables[0], ranked_tables[1]
    p1, p2 = profiles[t1], profiles[t2]

    d1 = _pick_time_col(p1)
    d2 = _pick_time_col(p2)
    m1 = _pick_measure_col(p1, question)
    m2 = _pick_measure_col(p2, question)

    if not d1 or not d2 or not m1 or not m2:
        return None

    return dedent(
        f"""
        WITH a AS (
          SELECT CAST({_qi(d1)} AS VARCHAR) AS period, SUM(COALESCE({_qi(m1)}, 0)) AS metric_a
          FROM {_qi(t1)}
          WHERE {_qi(d1)} IS NOT NULL
          GROUP BY 1
        ), b AS (
          SELECT CAST({_qi(d2)} AS VARCHAR) AS period, SUM(COALESCE({_qi(m2)}, 0)) AS metric_b
          FROM {_qi(t2)}
          WHERE {_qi(d2)} IS NOT NULL
          GROUP BY 1
        )
        SELECT
          COALESCE(a.period, b.period) AS period,
          a.metric_a,
          b.metric_b
        FROM a
        FULL OUTER JOIN b ON a.period = b.period
        ORDER BY period
        """
    ).strip()


def _build_dynamic_sql(question: str) -> tuple[str, list[str]]:
    wh = get_warehouse()
    q = question.lower()
    ranked = wh.rank_tables(question, top_n=4)

    if not ranked:
        raise ValueError("No tables available for SQL generation")

    profiles = wh.table_profiles()

    # Cross-table trend compare
    if "compare" in q and _contains_any(q, ["trend", "month", "over time", "time"]) and len(ranked) >= 2:
        selected = ranked[:2]
        if "domestic" in q and "international" in q:
            domestic = next(
                (
                    t
                    for t in ranked
                    if any(k in c.lower() for c in profiles[t].columns for k in ["state", "city", "ship"])
                ),
                None,
            )
            international = next(
                (
                    t
                    for t in ranked
                    if "international" in t.lower()
                    or any(k in c.lower() for c in profiles[t].columns for k in ["gross", "customer", "pcs"])
                ),
                None,
            )
            if domestic and international and domestic != international:
                selected = [domestic, international]

        cross_sql = _build_cross_table_trend_sql(question, selected)
        if cross_sql:
            return cross_sql, selected

    table = ranked[0]
    profile = profiles[table]

    limit_n = _limit_from_question(question, default=10)
    want_top = _contains_any(q, ["top", "rank", "highest", "descending", "best"])
    want_trend = _contains_any(q, ["trend", "month", "date", "over time", "timeline", "visual", "chart", "plot"]) 
    want_avg = _contains_any(q, ["average", "avg", "mean"])
    want_share = _contains_any(q, ["share", "percent", "percentage", "%", "ratio", "rate"])

    dim_col = _pick_dimension_col(profile, question)
    time_col = _pick_time_col(profile)
    measure_col = _pick_measure_col(profile, question)
    second_measure = _pick_secondary_measure_col(profile, measure_col)

    if "cancel" in q:
        for t in ranked:
            p = profiles[t]
            if any("cancel" in c.lower() for c in p.columns):
                table = t
                profile = p
                dim_col = _pick_dimension_col(profile, question)
                time_col = _pick_time_col(profile)
                measure_col = _pick_measure_col(profile, question)
                second_measure = _pick_secondary_measure_col(profile, measure_col)
                break

    if "expense" in q and "particular" in q:
        for t in ranked:
            p = profiles[t]
            if any("expense_particular" in c.lower() for c in p.columns):
                table = t
                profile = p
                dim_col = _pick_dimension_col(profile, question)
                time_col = _pick_time_col(profile)
                measure_col = _pick_measure_col(profile, question)
                second_measure = _pick_secondary_measure_col(profile, measure_col)
                break

    cancel_col = _find_col(profile.columns, ["is_cancelled", "cancel"])
    if "cancel" in q and cancel_col and dim_col:
        sql = dedent(
            f"""
            SELECT
              {_qi(dim_col)} AS dimension,
              100.0 * SUM(CASE WHEN COALESCE({_qi(cancel_col)}, FALSE) THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS cancellation_rate,
              COUNT(*) AS total_records
            FROM {_qi(table)}
            WHERE {_qi(dim_col)} IS NOT NULL
            GROUP BY {_qi(dim_col)}
            ORDER BY cancellation_rate DESC
            LIMIT {limit_n}
            """
        ).strip()
        return sql, [table]

    if want_trend and time_col and measure_col:
        sql = dedent(
            f"""
            SELECT
              CAST({_qi(time_col)} AS VARCHAR) AS period,
              SUM(COALESCE({_qi(measure_col)}, 0)) AS metric_value,
              COUNT(*) AS total_records
            FROM {_qi(table)}
            WHERE {_qi(time_col)} IS NOT NULL
            GROUP BY 1
            ORDER BY 1
            """
        ).strip()
        return sql, [table]

    if want_top and dim_col and measure_col:
        agg = "AVG" if want_avg else "SUM"
        sql = dedent(
            f"""
            SELECT
              {_qi(dim_col)} AS dimension,
              {agg}(COALESCE({_qi(measure_col)}, 0)) AS metric_value,
              COUNT(*) AS total_records
            FROM {_qi(table)}
            WHERE {_qi(dim_col)} IS NOT NULL
            GROUP BY {_qi(dim_col)}
            ORDER BY metric_value DESC
            LIMIT {limit_n}
            """
        ).strip()
        return sql, [table]

    if want_share and dim_col and measure_col:
        sql = dedent(
            f"""
            WITH base AS (
              SELECT
                {_qi(dim_col)} AS dimension,
                SUM(COALESCE({_qi(measure_col)}, 0)) AS metric_value
              FROM {_qi(table)}
              WHERE {_qi(dim_col)} IS NOT NULL
              GROUP BY {_qi(dim_col)}
            )
            SELECT
              dimension,
              metric_value,
              100.0 * metric_value / NULLIF((SELECT SUM(metric_value) FROM base), 0) AS metric_share_pct
            FROM base
            ORDER BY metric_share_pct DESC
            LIMIT {limit_n}
            """
        ).strip()
        return sql, [table]

    if _contains_any(q, ["compare", "difference", "delta"]) and dim_col and measure_col and second_measure:
        sql = dedent(
            f"""
            SELECT
              {_qi(dim_col)} AS dimension,
              AVG(COALESCE({_qi(measure_col)}, 0)) AS metric_a,
              AVG(COALESCE({_qi(second_measure)}, 0)) AS metric_b,
              AVG(COALESCE({_qi(measure_col)}, 0) - COALESCE({_qi(second_measure)}, 0)) AS metric_delta
            FROM {_qi(table)}
            WHERE {_qi(dim_col)} IS NOT NULL
            GROUP BY {_qi(dim_col)}
            ORDER BY ABS(metric_delta) DESC
            LIMIT {limit_n}
            """
        ).strip()
        return sql, [table]

    if measure_col:
        sql = dedent(
            f"""
            SELECT
              COUNT(*) AS total_records,
              SUM(COALESCE({_qi(measure_col)}, 0)) AS total_metric,
              AVG(COALESCE({_qi(measure_col)}, 0)) AS avg_metric
            FROM {_qi(table)}
            """
        ).strip()
        return sql, [table]

    sql = dedent(
        f"""
        SELECT *
        FROM {_qi(table)}
        LIMIT {limit_n}
        """
    ).strip()
    return sql, [table]


def _required_tables(question: str) -> list[str]:
    return get_warehouse().rank_tables(question, top_n=2)


def _heuristic_sql_validation(question: str, sql: str) -> tuple[bool, list[str]]:
    q = question.lower()
    s = sql.lower()
    notes: list[str] = []
    approved = True

    required = _required_tables(question)
    if required and not any(t.lower() in s for t in required):
        approved = False
        notes.append(f"SQL should reference one of top relevant tables: {', '.join(required)}")

    if _contains_any(q, ["top", "rank", "highest", "descending"]) and "order by" not in s:
        approved = False
        notes.append("Ranking query should include ORDER BY.")

    if _contains_any(q, ["top", "rank", "highest"]) and "limit" not in s:
        approved = False
        notes.append("Top/ranking query should include LIMIT.")

    if _contains_any(q, ["trend", "month", "date", "over time"]) and not any(k in s for k in ["date", "month", "period", "time"]):
        approved = False
        notes.append("Trend/time query should include time axis column.")

    if "cancel" in q and "cancel" not in s:
        approved = False
        notes.append("Cancellation query should include cancellation logic/field.")

    return approved, notes


def _parse_validator_sql_output(text: str) -> tuple[str, str, list[str]]:
    status_match = re.search(r"STATUS:\s*(APPROVED|REVISE)", text, flags=re.IGNORECASE)
    status = status_match.group(1).upper() if status_match else "REVISE"

    notes = []
    for line in text.splitlines():
        if line.lower().startswith("notes:"):
            notes.append(line.split(":", 1)[1].strip())

    sql = _extract_sql(text)
    return status, sql, notes


def _query_validator_agent(question: str, sql: str, issues: list[str], model_name: str) -> tuple[str, list[str], str]:
    try:
        llm = _build_llm(model_name)
        validator = Agent(
            role="SQL Validation Agent",
            goal="Validate SQL relevance and correctness against current schema and improve if required.",
            backstory="You are strict about semantic alignment and numerical correctness.",
            verbose=False,
            allow_delegation=False,
            llm=llm,
            tools=[get_dataset_playbook, get_relevant_sales_schema],
        )

        task = Task(
            description=dedent(
                f"""
                User question: {question}
                Candidate SQL:
                ```sql
                {sql}
                ```

                Heuristic issues:
                {chr(10).join(f"- {i}" for i in issues) if issues else '- none'}

                Validate query intent, schema usage, and metric definitions.
                Output strictly:
                STATUS: APPROVED or REVISE
                NOTES: <short reason>
                ```sql
                <final SQL>
                ```
                """
            ),
            expected_output="Approval status and final SQL.",
            agent=validator,
        )

        crew = Crew(agents=[validator], tasks=[task], process=Process.sequential, verbose=False)
        raw = str(crew.kickoff())
        status, candidate_sql, notes = _parse_validator_sql_output(raw)
        return (candidate_sql or sql), notes, status
    except Exception as exc:
        return sql, [f"Validator agent skipped ({exc})."], "APPROVED"


def _result_quality_checks(question: str, df: pd.DataFrame) -> tuple[bool, list[str]]:
    q = question.lower()
    notes: list[str] = []
    passed = True

    if df.empty:
        passed = False
        notes.append("Query returned empty result set.")

    if _contains_any(q, ["top", "rank", "highest"]) and not df.empty and len(df) < 3:
        passed = False
        notes.append("Top-N intent returned too few rows.")

    if _contains_any(q, ["sales", "revenue", "amount", "rate", "aov", "stock", "mrp", "tp", "cost"]) and not df.empty:
        has_numeric = any(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        if not has_numeric:
            passed = False
            notes.append("Metric-oriented query returned no numeric columns.")

    if _contains_any(q, ["trend", "month", "date", "over time"]) and not df.empty:
        has_time = any(any(k in c.lower() for k in ["date", "month", "period", "time"]) for c in df.columns)
        if not has_time:
            passed = False
            notes.append("Trend query returned no time-axis column.")

    return passed, notes


def _repair_sql(question: str, sql: str, failure_notes: list[str], mode: str, model_name: str) -> tuple[str, str]:
    fallback_sql, _ = _build_dynamic_sql(question)
    if fallback_sql and fallback_sql.lower() != sql.lower():
        return fallback_sql, "template_repair"

    if mode == "deep":
        revised, _, status = _query_validator_agent(
            question=question,
            sql=sql,
            issues=[f"Result quality failure: {n}" for n in failure_notes],
            model_name=model_name,
        )
        if status in {"APPROVED", "REVISE"} and revised:
            return revised, "agent_repair"

    return sql, "no_repair"


def _build_visualization_spec(question: str, df: pd.DataFrame) -> dict[str, Any]:
    q = question.lower()
    required = _contains_any(q, ["visual", "chart", "plot", "graph", "trend"])

    if df.empty:
        return {"required": False, "status": "skipped", "chart_type": "none", "x": None, "y": [], "notes": ["No rows for visualization."]}

    date_cols = [c for c in df.columns if _contains_any(c.lower(), ["date", "month", "period", "time"]) ]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    if (required or _contains_any(q, ["trend", "over time"])) and date_cols and num_cols:
        return {"required": True, "status": "proposed", "chart_type": "line", "x": date_cols[0], "y": [num_cols[0]], "notes": ["Time-series chart selected."]}

    if required and cat_cols and num_cols:
        return {"required": True, "status": "proposed", "chart_type": "bar", "x": cat_cols[0], "y": [num_cols[0]], "notes": ["Category comparison chart selected."]}

    return {"required": False, "status": "skipped", "chart_type": "none", "x": None, "y": [], "notes": ["No visualization requested."]}


def _parse_viz_validator_output(text: str) -> dict[str, Any]:
    status = "VALID"
    chart_type = "none"
    x_col = None
    y_cols: list[str] = []
    notes: list[str] = []

    for line in text.splitlines():
        low = line.lower().strip()
        if low.startswith("status:"):
            status = line.split(":", 1)[1].strip().upper()
        elif low.startswith("chart_type:"):
            chart_type = line.split(":", 1)[1].strip().lower()
        elif low.startswith("x:"):
            val = line.split(":", 1)[1].strip()
            x_col = None if val.lower() == "none" else val
        elif low.startswith("y:"):
            val = line.split(":", 1)[1].strip()
            y_cols = [] if val.lower() == "none" else [v.strip() for v in val.split(",") if v.strip()]
        elif low.startswith("notes:"):
            notes.append(line.split(":", 1)[1].strip())

    return {"status": status, "chart_type": chart_type, "x": x_col, "y": y_cols, "notes": notes}


def _visualization_validator_agent(question: str, df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    try:
        llm = _build_llm(model_name)
        validator = Agent(
            role="Visualization Validation Agent",
            goal="Validate chart necessity and mapping correctness against result data.",
            backstory="You ensure visuals are accurate and non-misleading.",
            verbose=False,
            allow_delegation=False,
            llm=llm,
        )

        task = Task(
            description=dedent(
                f"""
                Question: {question}
                Columns: {list(df.columns)}
                Rows: {len(df)}
                Sample: {df.head(5).to_dict(orient='records')}

                Decide chart mapping or skip.
                Output strictly:
                STATUS: VALID or REMAKE or SKIP
                CHART_TYPE: line or bar or none
                X: <column or none>
                Y: <comma-separated columns or none>
                NOTES: <short reason>
                """
            ),
            expected_output="Visualization validation specification.",
            agent=validator,
        )

        crew = Crew(agents=[validator], tasks=[task], process=Process.sequential, verbose=False)
        raw = str(crew.kickoff())
        return _parse_viz_validator_output(raw)
    except Exception as exc:
        return {"status": "SKIP", "chart_type": "none", "x": None, "y": [], "notes": [f"Visualization validator skipped ({exc})."]}


def _validate_visualization_spec(spec: dict[str, Any], df: pd.DataFrame, question: str, mode: str, model_name: str) -> dict[str, Any]:
    if not spec.get("required"):
        return spec

    notes = list(spec.get("notes", []))
    x_col = spec.get("x")
    y_cols = spec.get("y", [])

    def is_valid(x: Any, ys: list[Any]) -> bool:
        if not x or x not in df.columns or not ys:
            return False
        for c in ys:
            if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
                return False
        return True

    valid = is_valid(x_col, y_cols)

    if mode == "deep":
        agent_spec = _visualization_validator_agent(question=question, df=df, model_name=model_name)
        if agent_spec.get("status") in {"VALID", "REMAKE"} and is_valid(agent_spec.get("x"), agent_spec.get("y", [])):
            spec["chart_type"] = agent_spec.get("chart_type", spec.get("chart_type", "line"))
            spec["x"] = agent_spec.get("x")
            spec["y"] = agent_spec.get("y", [])
            spec["status"] = "validated_by_agent"
            notes.extend(agent_spec.get("notes", []))
            spec["notes"] = notes
            return spec

    if not valid:
        date_cols = [c for c in df.columns if _contains_any(c.lower(), ["date", "month", "period", "time"])]
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]

        if date_cols and num_cols:
            spec["chart_type"] = "line"
            spec["x"] = date_cols[0]
            spec["y"] = [num_cols[0]]
            spec["status"] = "auto_corrected"
            notes.append("Viz remapped to valid time + metric columns.")
        elif cat_cols and num_cols:
            spec["chart_type"] = "bar"
            spec["x"] = cat_cols[0]
            spec["y"] = [num_cols[0]]
            spec["status"] = "auto_corrected"
            notes.append("Viz remapped to valid category + metric columns.")
        else:
            spec["required"] = False
            spec["chart_type"] = "none"
            spec["x"] = None
            spec["y"] = []
            spec["status"] = "skipped"
            notes.append("No valid columns for chart rendering.")
    else:
        spec["status"] = "validated"

    spec["notes"] = notes
    return spec


def build_sql_generation_crew(user_question: str, model_name: str, mode: str) -> Crew:
    llm = _build_llm(model_name)

    generator = Agent(
        role="Adaptive SQL Generator",
        goal="Generate one efficient and correct SQL query from live schema context.",
        backstory="You are schema-driven, avoid assumptions, and optimize for correctness and speed.",
        verbose=False,
        allow_delegation=False,
        llm=llm,
        tools=[get_dataset_playbook, get_relevant_sales_schema],
    )

    complexity_note = "Prefer minimal query plan." if mode == "fast" else "Handle ambiguous intents with robust aggregation/filtering."

    task = Task(
        description=dedent(
            f"""
            User question: {user_question}

            Instructions:
            1) Call `Get dataset playbook` and `Get relevant sales schema` first.
            2) Use only tables/columns that exist in provided schema context.
            3) Generate one read-only SQL query (SELECT/WITH only).
            4) Include GROUP BY/ORDER BY/LIMIT when question implies ranking or top results.
            5) Include explicit date/month axis for trend questions.
            6) {complexity_note}

            Output strictly:
            ```sql
            <query>
            ```
            """
        ),
        expected_output="Single SQL query in SQL block.",
        agent=generator,
    )

    return Crew(agents=[generator], tasks=[task], process=Process.sequential, verbose=False)


@lru_cache(maxsize=256)
def _cached_query(normalized_question: str, mode: str, model_name: str) -> tuple[str, str]:
    # Fast path: deterministic adaptive SQL from schema profile.
    if mode == "fast":
        sql, _ = _build_dynamic_sql(normalized_question)
        return sql, "adaptive_fast_sql"

    # Deep path: agent first, adaptive fallback.
    try:
        crew = build_sql_generation_crew(normalized_question, model_name, mode)
        raw = str(crew.kickoff())
        sql = _extract_sql(raw)
        if sql:
            return sql, "llm_sql"
    except Exception:
        pass

    sql, _ = _build_dynamic_sql(normalized_question)
    return sql, "adaptive_fallback_sql"


def _execute_sql_with_validation(question: str, sql: str, mode: str, model_name: str, source: str) -> tuple[str, pd.DataFrame, str, list[str], str]:
    notes: list[str] = []
    is_valid, heuristic_notes = _heuristic_sql_validation(question, sql)
    notes.extend(heuristic_notes)

    final_sql = sql
    status = "approved"

    if mode == "deep":
        revised_sql, agent_notes, agent_status = _query_validator_agent(
            question=question,
            sql=final_sql,
            issues=heuristic_notes,
            model_name=model_name,
        )
        notes.extend(agent_notes)
        if revised_sql:
            final_sql = revised_sql
        status = "approved" if agent_status == "APPROVED" else "revised"
    elif not is_valid:
        corrected_sql, _ = _build_dynamic_sql(question)
        if corrected_sql.lower() != final_sql.lower():
            final_sql = corrected_sql
            status = "revised"
            notes.append("Applied adaptive fast-mode SQL correction.")

    max_rows = int(os.getenv("QUERY_MAX_ROWS", "200"))
    try:
        df = get_warehouse().execute_sql(final_sql, max_rows=max_rows)
    except Exception as exc:
        notes.append(f"Primary SQL failed: {exc}")
        df = pd.DataFrame()

    ok, quality_notes = _result_quality_checks(question, df)
    notes.extend(quality_notes)

    final_source = source

    if not ok:
        repair_sql, repair_source = _repair_sql(question=question, sql=final_sql, failure_notes=quality_notes, mode=mode, model_name=model_name)
        if repair_source != "no_repair":
            try:
                repaired_df = get_warehouse().execute_sql(repair_sql, max_rows=max_rows)
            except Exception as exc:
                notes.append(f"Repair SQL failed: {exc}")
                repaired_df = pd.DataFrame()

            repaired_ok, repaired_notes = _result_quality_checks(question, repaired_df)
            notes.append(f"Repair path: {repair_source}")
            notes.extend(repaired_notes)

            if repaired_ok or (not repaired_df.empty and df.empty):
                final_sql = repair_sql
                df = repaired_df
                final_source = f"{source}+{repair_source}"
                status = "repaired"

    return final_sql, df, status, notes, final_source


def _build_dataset_overview_result(mode: str, elapsed_seconds: float) -> PipelineResult:
    wh = get_warehouse()
    rows = []
    for p in wh.table_profiles().values():
        rows.append(
            {
                "table_name": p.table_name,
                "rows": p.row_count,
                "num_columns": len(p.columns),
                "numeric_columns": len(p.numeric_cols),
                "date_columns": len(p.date_cols),
            }
        )

    insights = [
        "Table profiling is automatic for all loaded CSV-derived tables/views.",
        "New CSV files become available as `sales_*` plus normalized `mart_dyn_*` views.",
        "Fast mode uses metadata-driven SQL synthesis without hardcoded file paths.",
        "Deep mode uses agents with schema context and validator checks before execution.",
        "Visualization is validated and auto-corrected based on result columns.",
    ]

    return PipelineResult(
        sql="-- dataset overview route",
        table_records=rows,
        table_columns=["table_name", "rows", "num_columns", "numeric_columns", "date_columns"],
        insights=insights,
        next_question="Do you want to run the full regression suite and review the latest HTML report?",
        elapsed_seconds=elapsed_seconds,
        mode=mode,
        source="dataset_overview",
        validation_status="approved",
        validation_notes=["Overview uses runtime metadata, no SQL generation required."],
        visualization={"required": False, "status": "skipped", "notes": ["No chart required."]},
    )


def answer_sales_question(user_question: str, mode: str = "fast") -> PipelineResult:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    normalized = _normalize_question(user_question)

    if not normalized:
        return PipelineResult(
            sql="",
            table_records=[],
            table_columns=[],
            insights=["Please enter a question."],
            next_question="",
            elapsed_seconds=0.0,
            mode=mode,
            source="input_validation",
            validation_status="rejected",
            validation_notes=["Empty question."],
            visualization={"required": False, "status": "skipped", "notes": ["No prompt provided."]},
        )

    start = time.perf_counter()

    if "insight" in normalized and ("each file" in normalized or "every file" in normalized or "all file" in normalized):
        result = _build_dataset_overview_result(mode=mode, elapsed_seconds=0.0)
        result.elapsed_seconds = time.perf_counter() - start
        return result

    base_sql, source = _cached_query(normalized, mode, model_name)

    final_sql, df, validation_status, validation_notes, final_source = _execute_sql_with_validation(
        question=normalized,
        sql=base_sql,
        mode=mode,
        model_name=model_name,
        source=source,
    )

    visualization = _build_visualization_spec(normalized, df)
    visualization = _validate_visualization_spec(
        spec=visualization,
        df=df,
        question=normalized,
        mode=mode,
        model_name=model_name,
    )

    records, columns = _table_to_payload(df)
    insights = _basic_insights(df, normalized)

    return PipelineResult(
        sql=final_sql,
        table_records=records,
        table_columns=columns,
        insights=insights,
        next_question=_suggest_next_question(normalized),
        elapsed_seconds=time.perf_counter() - start,
        mode=mode,
        source=final_source,
        validation_status=validation_status,
        validation_notes=validation_notes[:10],
        visualization=visualization,
    )
