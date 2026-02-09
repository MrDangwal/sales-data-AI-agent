from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List

import duckdb
import pandas as pd


SAFE_SQL_PATTERN = re.compile(r"^\s*(select|with)\b", flags=re.IGNORECASE | re.DOTALL)
FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|copy|attach|detach|call|pragma)\b",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NUMERIC_TYPE_PATTERN = re.compile(r"(tinyint|smallint|int|bigint|hugeint|decimal|double|float|real)", flags=re.IGNORECASE)
DATE_TYPE_PATTERN = re.compile(r"(date|timestamp|time)", flags=re.IGNORECASE)
BOOL_TYPE_PATTERN = re.compile(r"bool", flags=re.IGNORECASE)


@dataclass
class TableInfo:
    table_name: str
    file_name: str
    row_count: int
    columns: List[str]


@dataclass
class TableProfile:
    table_name: str
    row_count: int
    columns: List[str]
    numeric_cols: List[str]
    date_cols: List[str]
    text_cols: List[str]
    bool_cols: List[str]


class SalesWarehouse:
    def __init__(self, dataset_dir: str | Path, db_path: str | Path = ".cache/sales.duckdb") -> None:
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_mode = "persistent"
        try:
            self.conn = duckdb.connect(database=str(self.db_path))
        except Exception as exc:
            if "Conflicting lock is held" in str(exc):
                self.conn = duckdb.connect(database=":memory:")
                self.db_mode = "in_memory_fallback"
            else:
                raise

        self.conn.execute("PRAGMA threads=4")
        self.table_map: Dict[str, TableInfo] = {}
        self.profile_map: Dict[str, TableProfile] = {}
        self.schema_chunks: List[str] = []

        self._initialize_database()
        self._load_metadata()

    def _sanitize_table_name(self, name: str) -> str:
        base = Path(name).stem.lower()
        base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        return f"sales_{base}"

    def _sanitize_column_name(self, name: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        if not value:
            value = "col"
        if value[0].isdigit():
            value = f"c_{value}"
        return value

    def _quote_ident(self, ident: str) -> str:
        return "\"" + ident.replace("\"", "\"\"") + "\""

    def _is_numeric_type(self, col_type: str) -> bool:
        return bool(NUMERIC_TYPE_PATTERN.search(col_type or ""))

    def _is_date_type(self, col_type: str) -> bool:
        return bool(DATE_TYPE_PATTERN.search(col_type or ""))

    def _is_bool_type(self, col_type: str) -> bool:
        return bool(BOOL_TYPE_PATTERN.search(col_type or ""))

    def _initialize_database(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _ingestion_meta (
                file_name TEXT PRIMARY KEY,
                mtime_ns BIGINT,
                size_bytes BIGINT
            )
            """
        )

        if self._needs_refresh():
            self._refresh_raw_tables()
            self._refresh_analytics_marts()
            self._refresh_dynamic_marts()
            self._update_ingestion_meta()
        else:
            self._refresh_analytics_marts()
            self._refresh_dynamic_marts()

    def _needs_refresh(self) -> bool:
        csv_files = sorted(self.dataset_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_dir}")

        db_rows = {
            row[0]: (int(row[1]), int(row[2]))
            for row in self.conn.execute(
                "SELECT file_name, mtime_ns, size_bytes FROM _ingestion_meta"
            ).fetchall()
        }

        if len(db_rows) != len(csv_files):
            return True

        for csv_file in csv_files:
            stat = csv_file.stat()
            if csv_file.name not in db_rows:
                return True
            if db_rows[csv_file.name] != (stat.st_mtime_ns, stat.st_size):
                return True

        sample_table = self._sanitize_table_name(csv_files[0].name)
        exists = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [sample_table],
        ).fetchone()[0]
        return int(exists) == 0

    def _refresh_raw_tables(self) -> None:
        for csv_file in sorted(self.dataset_dir.glob("*.csv")):
            table_name = self._sanitize_table_name(csv_file.name)
            escaped_path = str(csv_file).replace("'", "''")
            self.conn.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * FROM read_csv_auto('{escaped_path}', header=true, ignore_errors=true)"
            )

    def _raw_table_names(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        return {r[0] for r in rows}

    def _refresh_analytics_marts(self) -> None:
        # Optional high-quality marts for known high-value sources.
        if "sales_amazon_sale_report" in self._raw_table_names():
            self.conn.execute(
                """
                CREATE OR REPLACE TABLE mart_amazon_orders AS
                SELECT
                    CAST("Order ID" AS VARCHAR) AS order_id,
                    COALESCE(
                        TRY_CAST("Date" AS DATE),
                        TRY_STRPTIME(CAST("Date" AS VARCHAR), '%m-%d-%y'),
                        TRY_STRPTIME(CAST("Date" AS VARCHAR), '%d-%m-%y'),
                        TRY_STRPTIME(CAST("Date" AS VARCHAR), '%Y-%m-%d')
                    ) AS order_date,
                    UPPER(TRIM(CAST("ship-state" AS VARCHAR))) AS ship_state,
                    UPPER(TRIM(CAST(Status AS VARCHAR))) AS order_status,
                    UPPER(TRIM(CAST(Fulfilment AS VARCHAR))) AS fulfillment,
                    UPPER(TRIM(CAST("Sales Channel" AS VARCHAR))) AS sales_channel,
                    COALESCE(TRY_CAST(Qty AS DOUBLE), 0) AS qty,
                    COALESCE(TRY_CAST(Amount AS DOUBLE), 0) AS amount,
                    UPPER(TRIM(CAST(Category AS VARCHAR))) AS category,
                    UPPER(TRIM(CAST(Size AS VARCHAR))) AS size,
                    UPPER(TRIM(CAST("ship-city" AS VARCHAR))) AS ship_city,
                    CASE WHEN UPPER(TRIM(CAST(Status AS VARCHAR))) LIKE '%CANCEL%' THEN TRUE ELSE FALSE END AS is_cancelled,
                    STRFTIME(
                        COALESCE(
                            TRY_CAST("Date" AS DATE),
                            TRY_STRPTIME(CAST("Date" AS VARCHAR), '%m-%d-%y'),
                            TRY_STRPTIME(CAST("Date" AS VARCHAR), '%d-%m-%y'),
                            TRY_STRPTIME(CAST("Date" AS VARCHAR), '%Y-%m-%d')
                        ),
                        '%Y-%m'
                    ) AS order_month
                FROM sales_amazon_sale_report
                """
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mart_amazon_orders_state ON mart_amazon_orders(ship_state)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mart_amazon_orders_date ON mart_amazon_orders(order_date)")

    def _refresh_dynamic_marts(self) -> None:
        raw_tables = sorted(t for t in self._raw_table_names() if t.startswith("sales_"))

        for raw_table in raw_tables:
            suffix = raw_table[len("sales_") :]
            dyn_view = f"mart_dyn_{suffix}"

            desc_df = self.conn.execute(f"DESCRIBE {self._quote_ident(raw_table)}").df()
            if desc_df.empty:
                continue

            select_parts: list[str] = []
            used_aliases: set[str] = set()

            for _, row in desc_df.iterrows():
                original_col = str(row["column_name"])
                col_type = str(row["column_type"])
                alias = self._sanitize_column_name(original_col)
                base_alias = alias
                idx = 2
                while alias in used_aliases:
                    alias = f"{base_alias}_{idx}"
                    idx += 1
                used_aliases.add(alias)

                q_col = self._quote_ident(original_col)
                q_alias = self._quote_ident(alias)
                select_parts.append(f"{q_col} AS {q_alias}")

                # For text-like columns, add parsed numeric/date helper projections.
                if (not self._is_numeric_type(col_type)) and (not self._is_date_type(col_type)) and (not self._is_bool_type(col_type)):
                    num_alias = f"{alias}_num"
                    date_alias = f"{alias}_date"
                    if num_alias not in used_aliases:
                        used_aliases.add(num_alias)
                        select_parts.append(
                            f"TRY_CAST(REGEXP_EXTRACT(CAST({q_col} AS VARCHAR), '[-+]?[0-9]*\\.?[0-9]+', 0) AS DOUBLE) AS {self._quote_ident(num_alias)}"
                        )
                    if date_alias not in used_aliases:
                        used_aliases.add(date_alias)
                        select_parts.append(
                            "COALESCE("
                            f"TRY_CAST({q_col} AS DATE),"
                            f"TRY_STRPTIME(CAST({q_col} AS VARCHAR), '%m-%d-%y'),"
                            f"TRY_STRPTIME(CAST({q_col} AS VARCHAR), '%d-%m-%y'),"
                            f"TRY_STRPTIME(CAST({q_col} AS VARCHAR), '%Y-%m-%d')"
                            f") AS {self._quote_ident(date_alias)}"
                        )

            if select_parts:
                select_sql = ",\n                    ".join(select_parts)
                self.conn.execute(
                    f"""
                    CREATE OR REPLACE VIEW {self._quote_ident(dyn_view)} AS
                    SELECT
                        {select_sql}
                    FROM {self._quote_ident(raw_table)}
                    """
                )

    def _update_ingestion_meta(self) -> None:
        self.conn.execute("DELETE FROM _ingestion_meta")
        for csv_file in sorted(self.dataset_dir.glob("*.csv")):
            stat = csv_file.stat()
            self.conn.execute(
                "INSERT INTO _ingestion_meta(file_name, mtime_ns, size_bytes) VALUES (?, ?, ?)",
                [csv_file.name, stat.st_mtime_ns, stat.st_size],
            )

    def _build_profile(self, table_name: str, row_count: int) -> TableProfile:
        desc_df = self.conn.execute(f"DESCRIBE {self._quote_ident(table_name)}").df()
        columns = desc_df["column_name"].astype(str).tolist()
        numeric_cols: list[str] = []
        date_cols: list[str] = []
        bool_cols: list[str] = []
        text_cols: list[str] = []

        for _, row in desc_df.iterrows():
            name = str(row["column_name"])
            col_type = str(row["column_type"]) if row["column_type"] is not None else ""
            lname = name.lower()

            if self._is_numeric_type(col_type):
                numeric_cols.append(name)
            elif self._is_bool_type(col_type):
                bool_cols.append(name)
            elif self._is_date_type(col_type) or any(k in lname for k in ["date", "month", "time", "period", "year"]):
                date_cols.append(name)
                text_cols.append(name)
            else:
                text_cols.append(name)

        return TableProfile(
            table_name=table_name,
            row_count=row_count,
            columns=columns,
            numeric_cols=numeric_cols,
            date_cols=date_cols,
            text_cols=text_cols,
            bool_cols=bool_cols,
        )

    def _load_metadata(self) -> None:
        table_names = sorted(
            r[0]
            for r in self.conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema='main'
                  AND table_name NOT LIKE '\\_%' ESCAPE '\\'
                """
            ).fetchall()
        )

        self.table_map = {}
        self.profile_map = {}
        for table_name in table_names:
            row_count = int(self.conn.execute(f"SELECT COUNT(*) FROM {self._quote_ident(table_name)}").fetchone()[0])
            profile = self._build_profile(table_name=table_name, row_count=row_count)
            self.table_map[table_name] = TableInfo(
                table_name=table_name,
                file_name=f"{table_name}.table",
                row_count=row_count,
                columns=profile.columns,
            )
            self.profile_map[table_name] = profile

        self.schema_chunks = self._build_schema_chunks(max_chars=1000)

    def table_profiles(self) -> Dict[str, TableProfile]:
        return self.profile_map

    def _expand_keywords(self, question: str) -> set[str]:
        keywords = set(TOKEN_PATTERN.findall(question.lower()))
        expansion: dict[str, set[str]] = {
            "sales": {"revenue", "amount", "gross", "value"},
            "revenue": {"sales", "amount", "gross", "value"},
            "quantity": {"qty", "units", "pcs", "stock", "volume"},
            "qty": {"quantity", "units", "pcs", "stock"},
            "trend": {"month", "date", "time", "period"},
            "stock": {"inventory", "qty", "units"},
            "mrp": {"price", "tp", "rate"},
            "expense": {"cost", "spend", "amount"},
            "customer": {"buyer", "client"},
        }
        expanded = set(keywords)
        for token in list(keywords):
            expanded.update(expansion.get(token, set()))
        return expanded

    def _profile_score(self, profile: TableProfile, keywords: set[str]) -> int:
        table_tokens = set(TOKEN_PATTERN.findall(profile.table_name.lower()))
        col_tokens = set()
        for col in profile.columns:
            col_tokens.update(TOKEN_PATTERN.findall(col.lower()))

        score = len(table_tokens & keywords) * 3 + len(col_tokens & keywords)

        if profile.table_name.startswith("mart_dyn_"):
            score -= 1
        elif profile.table_name.startswith("mart_"):
            score += 3

        if any(k in keywords for k in {"trend", "month", "date", "time", "period"}) and profile.date_cols:
            score += 2
        if any(k in keywords for k in {"sales", "revenue", "amount", "value", "gross", "rate", "mrp", "tp", "stock"}) and profile.numeric_cols:
            score += 2

        lower_cols = [c.lower() for c in profile.columns]
        if "cancel" in keywords and any("cancel" in c for c in lower_cols):
            score += 4
        if "expense" in keywords and any("expense" in c for c in lower_cols):
            score += 4
        if "domestic" in keywords and any(k in c for c in lower_cols for k in ["state", "city", "country"]):
            score += 3
        if "international" in keywords and (
            "international" in profile.table_name.lower()
            or any(k in c for c in lower_cols for k in ["gross", "customer", "pcs"])
        ):
            score += 3

        return score

    def rank_tables(self, question: str, top_n: int = 5) -> List[str]:
        keywords = self._expand_keywords(question)
        scored: list[tuple[int, str]] = []
        for profile in self.profile_map.values():
            score = self._profile_score(profile, keywords)
            scored.append((score, profile.table_name))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        chosen = [name for score, name in scored if score > 0][:top_n]
        if chosen:
            return chosen

        fallback = [name for _, name in scored if name.startswith("mart_")]
        if fallback:
            return fallback[:top_n]
        return [name for _, name in scored[:top_n]]

    def semantic_catalog_text(self, question: str, top_n: int = 4) -> str:
        ranked = self.rank_tables(question, top_n=top_n)
        lines = ["Relevant table catalog:"]
        for table_name in ranked:
            p = self.profile_map[table_name]
            lines.append(f"- {table_name} (rows={p.row_count})")
            lines.append(f"  numeric: {', '.join(p.numeric_cols[:10]) if p.numeric_cols else '-'}")
            lines.append(f"  date/time: {', '.join(p.date_cols[:10]) if p.date_cols else '-'}")
            lines.append(f"  text/dim: {', '.join(p.text_cols[:12]) if p.text_cols else '-'}")
        return "\n".join(lines)

    def dataset_playbook(self) -> str:
        marts = sorted(name for name in self.table_map if name.startswith("mart_"))
        preview = ", ".join(marts[:12]) + (" ..." if len(marts) > 12 else "")
        return (
            "Dataset Playbook:\n"
            "- Route each question to the most relevant mart/table using column overlap with question terms.\n"
            "- Prefer marts over raw tables for cleaner schema and parsed helper columns.\n"
            "- For ranking questions: use GROUP BY + ORDER BY + LIMIT.\n"
            "- For trend questions: include a date/month/period axis and aggregate metrics.\n"
            "- For rate/share questions: use explicit numerator/denominator formulas with NULLIF for division safety.\n"
            f"- Available marts/views: {preview}\n"
            "- If new CSV files are added, corresponding `sales_*` and `mart_dyn_*` objects are created automatically."
        )

    def _build_schema_chunks(self, max_chars: int) -> List[str]:
        chunks: List[str] = []
        current = ""
        for info in self.table_map.values():
            block = (
                f"table={info.table_name} rows={info.row_count}\n"
                f"columns={', '.join(info.columns)}\n"
            )
            if current and len(current) + len(block) > max_chars:
                chunks.append(current.strip())
                current = block
            else:
                current += block
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def schema_text(self) -> str:
        lines = ["Loaded tables/views:"]
        for info in self.table_map.values():
            lines.append(f"- {info.table_name} (rows: {info.row_count})")
            lines.append(f"  columns: {', '.join(info.columns)}")
        return "\n".join(lines)

    def relevant_schema_text(self, question: str, max_chunks: int = 2) -> str:
        if not question.strip() or not self.table_map:
            return self.schema_text()

        ranked = self.rank_tables(question, top_n=max(1, max_chunks * 2))
        blocks = []
        for table_name in ranked:
            info = self.table_map[table_name]
            blocks.append(
                f"table={info.table_name} rows={info.row_count}\ncolumns={', '.join(info.columns)}"
            )
        return "Relevant schema context:\n" + "\n\n".join(blocks)

    def execute_sql(self, sql: str, max_rows: int = 200) -> pd.DataFrame:
        sql = sql.strip().rstrip(";")
        if not SAFE_SQL_PATTERN.search(sql) or FORBIDDEN_SQL_PATTERN.search(sql):
            raise ValueError("Only read-only SELECT/WITH queries are allowed.")

        limited_sql = f"SELECT * FROM ({sql}) t LIMIT {max_rows}"
        return self.conn.execute(limited_sql).df()

    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted = df.copy()
        for col in formatted.columns:
            if pd.api.types.is_numeric_dtype(formatted[col]):
                cname = str(col).lower()
                if any(k in cname for k in ["amount", "sales", "revenue", "value", "cost", "price", "gross", "mrp", "tp"]):
                    formatted[col] = formatted[col].map(lambda x: f"{float(x):,.2f}")
                elif "ratio" in cname or "rate" in cname or "pct" in cname or "share" in cname:
                    formatted[col] = formatted[col].map(lambda x: f"{float(x):,.2f}%")
                else:
                    formatted[col] = formatted[col].map(
                        lambda x: f"{float(x):,.0f}" if float(x).is_integer() else f"{float(x):,.2f}"
                    )
        return formatted

    def kpi_snapshot(self) -> str:
        if "mart_amazon_orders" in self.table_map:
            sql = """
            SELECT
                COUNT(*) AS total_orders,
                SUM(qty) AS total_units,
                SUM(amount) AS gross_revenue,
                AVG(amount) AS avg_order_value,
                100.0 * SUM(CASE WHEN is_cancelled THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS cancellation_rate
            FROM mart_amazon_orders
            """
            return self.format_dataframe(self.execute_sql(sql, max_rows=10)).to_markdown(index=False)

        ranked = self.rank_tables("sales revenue trend", top_n=1)
        if not ranked:
            return "No suitable tables found for KPI snapshot."

        table = ranked[0]
        profile = self.profile_map[table]
        if not profile.numeric_cols:
            return f"No numeric metrics found in {table} for KPI snapshot."

        metric = profile.numeric_cols[0]
        sql = f"SELECT COUNT(*) AS total_rows, SUM({self._quote_ident(metric)}) AS total_metric FROM {self._quote_ident(table)}"
        return self.format_dataframe(self.execute_sql(sql, max_rows=10)).to_markdown(index=False)


def create_warehouse(dataset_dir: str | Path = "Sales Dataset") -> SalesWarehouse:
    return SalesWarehouse(dataset_dir=dataset_dir)
