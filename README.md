# Sales GenAI Assignment - Production-Ready CrewAI + OpenAI

This repository contains a production-oriented implementation of a **multi-agent sales chatbot** and **sales summarization script** using **CrewAI**, **OpenAI**, and **DuckDB** over CSV files in `Sales Dataset/`.

## What Was Upgraded
- Lower latency path with **Fast mode** using adaptive schema-driven SQL synthesis
- Higher quality path with **Deep mode** using agent-assisted SQL generation + validation
- **Schema chunking + relevance retrieval** to reduce prompt size and token cost
- **Persistent DuckDB storage** (`.cache/sales.duckdb`) with incremental refresh checks
- **DuckDB-native ingestion** (`read_csv_auto`) for faster startup and lower memory than pandas loading
- **Automatic dynamic marts** (`mart_dyn_*`) generated for every newly added CSV table
- **Response caching** for repeated questions
- **Adaptive dataset playbook** driven by live table/column profiles
- **Validation agents** for SQL relevance and visualization correctness (with auto-repair path)
- Better Streamlit UI with structured result rendering (SQL code block + dataframe + insights)
- Query safety with SELECT-only SQL guard

## Deliverables Covered
- Working multi-agent chatbot (`app.py`)
- Working summarization script (`summarize_sales.py`)
- Runs on provided sample sales data (CSV)
- Setup/dependencies and technical notes
- Architecture presentation content (`docs/Architecture_Presentation.md`)
- Demo evidence templates (`demo/`)

## Project Structure
- `app.py`: Streamlit UI with Fast/Deep mode and structured output rendering
- `summarize_sales.py`: executive summary generation
- `src/sales_data.py`: persistent DB ingestion, dynamic marts, profiling, SQL execution, schema chunking
- `src/crew_pipeline.py`: adaptive SQL builder + validator agents + deep-mode LLM fallback
- `tests/test_sales_data.py`: core tests
- `docs/Architecture_Presentation.md`: slide content
- `docs/test_questions_by_file.md`: 5 test questions per dataset file
- `demo/example_qa.md`: sample Q&A
- `demo/example_summary_output.md`: sample summary

## Multi-Agent Architecture
- CrewAI agents (deep mode) are used to generate SQL grounded in:
  - `Get dataset playbook`
  - `Get relevant sales schema`
- Fast mode uses adaptive metadata-driven SQL generation (no fixed file routing).
- Validation layer:
  - SQL Validation Agent: checks query relevance and data-definition correctness
  - Result-quality checks: validates whether returned data is adequate, retries with repaired SQL if needed
  - Visualization Validation Agent: decides chart necessity, validates chart mappings, auto-corrects invalid charts
- Execution and presentation are handled in application code for consistency.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set environment variables:
```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
```

## Run Chatbot
```bash
streamlit run app.py
```

## Run Summarization
```bash
python summarize_sales.py
```

## Run Tests
```bash
pytest -q
```

## Run Regression Suite (CSV + HTML)
Execute all questions from `docs/test_questions_by_file.md` and generate one consolidated report:
```bash
python run_regression.py --mode fast --output-dir reports
```

Optional deep validation run:
```bash
python run_regression.py --mode deep --output-dir reports
```

Notes:
- `run_regression.py` auto-loads `.env` before execution.
- Use `--limit N` for a quick smoke run (example: `--limit 5`).
- Outputs are timestamped in `reports/` as `regression_report_<mode>_<timestamp>.csv/.html`.

## Runtime Controls (UI)
- Pipeline mode: `Fast` or `Deep`
- Model override (`OPENAI_MODEL`)

## Production Notes
- SQL is read-only (guards block DDL/DML)
- Existing curated marts are used when available, but new files are handled automatically through dynamic marts (`mart_dyn_*`)
- Fast mode keeps latency low by using adaptive rule-based generation from live schema profiles
- Deep mode adds validator agents and schema-grounded LLM refinement when needed
- Relevant schema chunking limits unnecessary context
- Caching reduces repeated inference latency
- DuckDB vectorized execution handles analytical queries efficiently

## Assumptions
- CSV headers remain mostly stable
- OpenAI credentials are available
- Single-process deployment for assignment scope

## Limitations
- No auth/RBAC yet
- No persistent warehouse/catalog metadata store
- No distributed tracing backend

## Next Improvements
1. Add semantic metric layer and governed business glossary
2. Add OpenTelemetry traces and centralized logs
3. Add async background workers and request queueing
4. Add Parquet/object storage + partitioned incremental ingestion

## GitHub Submission
```bash
git init
git add .
git commit -m "GenAI sales multi-agent production-ready assignment"
git branch -M main
git remote add origin <your_repo_url>
git push -u origin main
```
Share the GitHub repo URL in your submission.
