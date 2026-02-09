from __future__ import annotations

import os
from textwrap import dedent

from crewai import Agent, Crew, Process, Task

from src.sales_data import create_warehouse


def build_summary_prompt() -> str:
    warehouse = create_warehouse("Sales Dataset")
    schema_chunk = warehouse.relevant_schema_text(
        "executive summary revenue volume fulfillment cancellations by region", max_chunks=3
    )
    kpis = warehouse.kpi_snapshot()
    return dedent(
        f"""
        You are given a chunked schema context and KPI snapshot.

        Schema Context:
        {schema_chunk}

        KPI Snapshot:
        {kpis}

        Create an executive summary with:
        1) Data coverage overview
        2) Revenue and volume highlights
        3) Operational concerns (status/cancellations/fulfillment)
        4) 5 recommended next analyses

        Keep it concise and evidence-based.
        """
    )


def run_summary() -> str:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    summarizer = Agent(
        role="Executive Sales Summarizer",
        goal="Deliver an accurate, concise summary for leadership.",
        backstory="You are a principal analytics consultant focused on clarity and facts.",
        llm=model_name,
        verbose=False,
    )

    task = Task(
        description=build_summary_prompt(),
        expected_output="Executive summary in structured sections.",
        agent=summarizer,
    )

    crew = Crew(agents=[summarizer], tasks=[task], process=Process.sequential, verbose=False)
    return str(crew.kickoff())


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    print(run_summary())
