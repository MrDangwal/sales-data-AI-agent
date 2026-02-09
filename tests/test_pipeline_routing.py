from src.crew_pipeline import answer_sales_question


def test_cancellation_query_routes_to_amazon_mart():
    result = answer_sales_question(
        "Cancellation rate by fulfillment channel with total order counts.",
        mode="fast",
    )
    assert "amazon" in result.sql.lower()
    assert "fulfillment" in result.sql.lower()
    assert len(result.table_records) > 0


def test_expense_particular_query_routes_to_expense_mart():
    result = answer_sales_question(
        "Top expense particulars by amount.",
        mode="fast",
    )
    sql = result.sql.lower()
    assert "expense_iigf" in sql
    assert "particular" in sql
    assert len(result.table_records) > 1


def test_cross_file_trend_query_uses_both_tables():
    result = answer_sales_question(
        "Compare domestic sales trend with international gross amount trend.",
        mode="fast",
    )
    sql = result.sql.lower()
    assert "amazon" in sql
    assert "international" in sql
    assert len(result.table_records) > 0
