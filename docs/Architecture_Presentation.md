# Sales GenAI Assignment - Architecture Presentation

## Slide 1 - Problem and Objective
- Build a production-ready multi-agent analytics assistant over sales CSV corpora.
- Answer business questions with high accuracy, low latency, and explainable outputs.
- Support both chatbot interaction and regression-scale automated validation.

## Slide 2 - System Architecture and Data Flow
- UI Layer: Streamlit (`app.py`) for chat, result table, validation notes, and charts.
- Orchestration Layer: Adaptive pipeline (`src/crew_pipeline.py`) with fast and deep modes.
- Data Layer: DuckDB persistent storage (`.cache/sales.duckdb`) + dynamic marts.
- Validation Layer: SQL validator, result-quality checks, visualization validator.
- Flow: Question -> table ranking -> SQL generation -> validation/repair -> execution -> formatted response.

## Slide 3 - LLM Integration Strategy
- Fast mode: schema-driven deterministic SQL synthesis (minimal LLM dependency).
- Deep mode: CrewAI agent SQL generation + validator agents + adaptive fallback.
- Grounding tools:
  - Get dataset playbook
  - Get relevant schema/catalog
- Hallucination control: only SQL over discovered schema; read-only guard enforced.

## Slide 4 - Data Storage, Indexing, Retrieval (100GB Scale)
- Current: DuckDB over CSV with incremental refresh and dynamic marts.
- 100GB target:
  - Store in Parquet on object storage with partitioning (date/category/channel).
  - Query engine: DuckDB/Trino/Snowflake external tables.
  - Metadata catalog + semantic layer for governed metric definitions.
  - Retrieval: profile-based table ranking + schema chunks + query/result caching.
- Indexing approach:
  - Physical: partition pruning, clustered sort keys.
  - Logical: semantic tags on columns (metric, dimension, time).

## Slide 5 - Example Query-Response Pipeline
- Example: "Top 10 states by sales and cancellation rate"
1. Rank relevant tables from live schema profiles.
2. Generate adaptive SQL using best-fit dimensions/measures.
3. Validate SQL intent alignment and metric semantics.
4. Execute SQL in DuckDB, validate result adequacy.
5. Build final answer: SQL, table, insights, and optional validated chart.

## Slide 6 - Accuracy and Validation Design
- SQL validation checks:
  - table/column relevance
  - ranking/trend syntax expectations
  - cancellation/rate semantic checks
- Result validation checks:
  - empty result handling
  - metric presence checks
  - intent-to-shape checks (top/trend/share)
- Visualization checks:
  - whether chart is needed
  - valid x/y mapping
  - auto-remap or skip if invalid

## Slide 7 - Cost and Performance Considerations
- Latency controls:
  - fast mode avoids unnecessary LLM calls
  - caching of normalized question-to-SQL mapping
  - DuckDB vectorized execution
- Cost controls:
  - deep mode only when needed
  - bounded output rows and compact prompts
  - regression automation to detect quality drift early

## Slide 8 - Operational Readiness and Deliverables
- Automated regression runner (`run_regression.py`) exports CSV + HTML reports.
- Adaptive architecture handles new CSV files without hardcoding.
- Current benchmark (fast mode): full 40-question suite with 0 errors and 0 empty results.
- Submission artifacts:
  - codebase
  - README/setup
  - screenshots/demo
  - this architecture deck (PDF)
