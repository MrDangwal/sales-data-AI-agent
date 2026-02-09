from run_regression import parse_question_bank


def test_parse_question_bank_count_and_sections():
    items = parse_question_bank("docs/test_questions_by_file.md")
    assert len(items) == 40
    assert any("Amazon Sale Report" in i.section for i in items)
    assert any("Cross-File Validation Questions" in i.section for i in items)


def test_parse_question_bank_question_text():
    items = parse_question_bank("docs/test_questions_by_file.md")
    assert any("Top 10 states by total sales and total units." == i.question for i in items)
