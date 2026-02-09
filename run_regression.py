from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
import traceback

import pandas as pd
from dotenv import load_dotenv

from src.crew_pipeline import answer_sales_question


QUESTION_LINE_RE = r"^\d+\.\s+(.*)$"
SECTION_LINE_RE = r"^##\s+(.*)$"


@dataclass
class QuestionItem:
    section: str
    question: str


def parse_question_bank(question_file: str | Path) -> list[QuestionItem]:
    path = Path(question_file)
    if not path.exists():
        raise FileNotFoundError(f"Question bank not found: {path}")

    import re

    section = "Uncategorized"
    items: list[QuestionItem] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = re.match(SECTION_LINE_RE, line)
        if section_match:
            section = section_match.group(1).strip()
            continue

        question_match = re.match(QUESTION_LINE_RE, line)
        if question_match:
            question = question_match.group(1).strip()
            items.append(QuestionItem(section=section, question=question))

    return items


def _build_html_report(df: pd.DataFrame, summary: dict[str, object], report_title: str) -> str:
    summary_df = pd.DataFrame([summary])
    table_html = df.to_html(index=False, escape=True)
    summary_html = summary_df.to_html(index=False, escape=True)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{report_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #6b7280; margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    .section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>{report_title}</h1>
  <p class=\"muted\">Generated at {datetime.now().isoformat(timespec='seconds')}</p>

  <div class=\"section\">
    <h2>Summary</h2>
    {summary_html}
  </div>

  <div class=\"section\">
    <h2>Detailed Results</h2>
    {table_html}
  </div>
</body>
</html>
"""


def run_regression(
    question_file: str | Path,
    mode: str,
    output_dir: str | Path,
    limit: int | None = None,
) -> tuple[Path, Path, dict[str, object], pd.DataFrame]:
    questions = parse_question_bank(question_file)
    if not questions:
        raise ValueError("No questions parsed from question file.")

    if limit is not None and limit > 0:
        questions = questions[:limit]

    rows: list[dict[str, object]] = []
    for idx, item in enumerate(questions, start=1):
        run_start = perf_counter()
        try:
            result = answer_sales_question(item.question, mode=mode)
            runner_elapsed = perf_counter() - run_start

            viz = result.visualization or {}
            y_cols = viz.get("y", [])
            if not isinstance(y_cols, list):
                y_cols = [str(y_cols)]

            rows.append(
                {
                    "index": idx,
                    "section": item.section,
                    "question": item.question,
                    "mode": result.mode,
                    "source": result.source,
                    "validation_status": result.validation_status,
                    "result_rows": len(result.table_records),
                    "result_columns": ", ".join(result.table_columns),
                    "sql": result.sql,
                    "insights": " | ".join(result.insights),
                    "next_question": result.next_question,
                    "validation_notes": " | ".join(result.validation_notes),
                    "viz_required": bool(viz.get("required", False)),
                    "viz_status": str(viz.get("status", "")),
                    "viz_chart_type": str(viz.get("chart_type", "")),
                    "viz_x": str(viz.get("x", "")),
                    "viz_y": ", ".join(map(str, y_cols)),
                    "pipeline_elapsed_seconds": round(float(result.elapsed_seconds), 3),
                    "runner_elapsed_seconds": round(float(runner_elapsed), 3),
                    "error": "",
                }
            )
        except Exception as exc:  # keep run going across failures
            runner_elapsed = perf_counter() - run_start
            rows.append(
                {
                    "index": idx,
                    "section": item.section,
                    "question": item.question,
                    "mode": mode,
                    "source": "error",
                    "validation_status": "error",
                    "result_rows": 0,
                    "result_columns": "",
                    "sql": "",
                    "insights": "",
                    "next_question": "",
                    "validation_notes": "",
                    "viz_required": False,
                    "viz_status": "error",
                    "viz_chart_type": "",
                    "viz_x": "",
                    "viz_y": "",
                    "pipeline_elapsed_seconds": 0.0,
                    "runner_elapsed_seconds": round(float(runner_elapsed), 3),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    df = pd.DataFrame(rows)
    total = len(df)
    error_count = int((df["error"] != "").sum()) if "error" in df.columns else 0
    empty_count = int((df["result_rows"] == 0).sum()) if "result_rows" in df.columns else 0
    approved_count = int(df["validation_status"].isin(["approved", "repaired", "revised"]).sum()) if "validation_status" in df.columns else 0
    avg_latency = float(df["runner_elapsed_seconds"].mean()) if "runner_elapsed_seconds" in df.columns else 0.0

    summary = {
        "total_questions": total,
        "approved_or_repaired": approved_count,
        "errors": error_count,
        "empty_results": empty_count,
        "avg_runner_latency_s": round(avg_latency, 3),
        "mode": mode,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"regression_report_{mode}_{ts}.csv"
    html_path = out_dir / f"regression_report_{mode}_{ts}.html"

    df.to_csv(csv_path, index=False)
    html_path.write_text(
        _build_html_report(df=df, summary=summary, report_title=f"Sales Regression Report ({mode})"),
        encoding="utf-8",
    )

    return csv_path, html_path, summary, df


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run regression evaluation using markdown question bank.")
    parser.add_argument(
        "--question-file",
        default="docs/test_questions_by_file.md",
        help="Path to markdown question bank file.",
    )
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "deep"],
        help="Pipeline mode to execute.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write report artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for first N questions.",
    )

    args = parser.parse_args()

    csv_path, html_path, summary, _ = run_regression(
        question_file=args.question_file,
        mode=args.mode,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    print("Regression run complete")
    print(f"CSV: {csv_path}")
    print(f"HTML: {html_path}")
    print("Summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
