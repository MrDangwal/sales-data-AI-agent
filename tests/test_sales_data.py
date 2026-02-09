from src.sales_data import create_warehouse


def test_schema_has_tables():
    wh = create_warehouse("Sales Dataset")
    schema = wh.schema_text()
    assert "mart_amazon_orders" in schema


def test_read_only_sql_guard():
    wh = create_warehouse("Sales Dataset")
    try:
        wh.execute_sql("DROP TABLE mart_amazon_orders")
        assert False, "Expected ValueError for unsafe SQL"
    except ValueError:
        assert True


def test_simple_query_runs():
    wh = create_warehouse("Sales Dataset")
    df = wh.execute_sql("SELECT COUNT(*) AS c FROM mart_amazon_orders")
    assert int(df.iloc[0]["c"]) > 0


def test_cancellation_field_exists():
    wh = create_warehouse("Sales Dataset")
    df = wh.execute_sql("SELECT COUNT(*) AS c FROM mart_amazon_orders WHERE is_cancelled")
    assert int(df.iloc[0]["c"]) >= 0


def test_order_date_parsing_is_available():
    wh = create_warehouse("Sales Dataset")
    df = wh.execute_sql("SELECT COUNT(*) AS c FROM mart_amazon_orders WHERE order_date IS NOT NULL")
    assert int(df.iloc[0]["c"]) > 0


def test_relevant_schema_chunking():
    wh = create_warehouse("Sales Dataset")
    txt = wh.relevant_schema_text("amazon amount qty by state", max_chunks=1).lower()
    assert "mart_amazon_orders" in txt


def test_cte_sql_is_allowed():
    wh = create_warehouse("Sales Dataset")
    df = wh.execute_sql(
        """
        WITH t AS (
            SELECT COUNT(*) AS c FROM mart_amazon_orders
        )
        SELECT c FROM t
        """
    )
    assert int(df.iloc[0]["c"]) > 0
