from theories_pipeline.outputs import QUESTION_COLUMNS
from theories_pipeline.result_tables import (
    normalize_answer,
    prepare_collected_papers,
    prepare_normalised_answers,
    prepare_theory_table,
)


def test_normalize_answer_variants():
    assert (
        normalize_answer("Q1", "Yes, quantitatively shown - biomarker details")
        == "Yes, quantitatively shown"
    )
    assert normalize_answer("Q1", "Yes, mentioned without data") == "Yes, but not shown"
    assert normalize_answer("Q1", "No evidence found") == "No"
    assert normalize_answer("Q2", "Mechanism supported by experiments") == "Yes"
    assert normalize_answer("Q2", "No mechanism discussed") == "No"
    assert normalize_answer("Q4", "Not discussed - insufficient data") == "No"
    assert normalize_answer("Q6", "Primary focus of the paper") == "Yes"
    assert normalize_answer("Q9", "Observational evidence presented") == "Yes"
    # Unknown answers are returned trimmed for downstream inspection
    assert normalize_answer("Q3", "Inconclusive") == "Inconclusive"


def test_prepare_theory_table_sorts_and_deduplicates():
    rows = [
        {"theory_id": "theory-b", "theory_name": "Beta", "number_of_collected_papers": "2"},
        {"theory_id": "theory-a", "theory_name": "Alpha", "number_of_collected_papers": "5"},
        # Duplicate entry with a worse count should be ignored
        {"theory_id": "theory-a", "theory_name": "Alpha", "number_of_collected_papers": "3"},
        {"theory_id": "theory-c", "theory_name": "Gamma", "number_of_collected_papers": ""},
    ]

    result = prepare_theory_table(rows)
    assert result == [
        {"theory_id": "theory-a", "theory_name": "Alpha", "number_of_collected_papers": "5"},
        {"theory_id": "theory-b", "theory_name": "Beta", "number_of_collected_papers": "2"},
        {"theory_id": "theory-c", "theory_name": "Gamma", "number_of_collected_papers": "0"},
    ]


def test_prepare_collected_papers_deduplicates_and_sorts():
    rows = [
        {
            "theory_id": "T2",
            "paper_url": "paper-2",
            "paper_name": "Beta study",
            "paper_year": "2021",
        },
        {
            "theory_id": "T1",
            "paper_url": "paper-1",
            "paper_name": "Alpha study",
            "paper_year": "2020",
        },
        {
            "theory_id": "T1",
            "paper_url": "paper-1",
            "paper_name": "Alpha study",
            "paper_year": "2020",
        },
        {"theory_id": "T3", "paper_url": "", "paper_name": "", "paper_year": ""},
    ]

    result = prepare_collected_papers(rows)
    assert result == [
        {
            "theory_id": "T1",
            "paper_url": "paper-1",
            "paper_name": "Alpha study",
            "paper_year": "2020",
        },
        {
            "theory_id": "T2",
            "paper_url": "paper-2",
            "paper_name": "Beta study",
            "paper_year": "2021",
        },
    ]


def test_prepare_normalised_answers_orders_and_normalises():
    rows = [
        {
            "theory_id": "T2",
            "paper_url": "paper-2",
            "paper_name": "Beta study",
            "paper_year": "2021",
            "Q1": "No evidence found",
        },
        {
            "theory_id": "T1",
            "paper_url": "paper-1",
            "paper_name": "Alpha study",
            "paper_year": "2020",
            "Q1": "Yes, mentioned without data - summary",
            "Q2": "Mechanism supported by experiments",
        },
    ]

    result = prepare_normalised_answers(rows)
    assert [row["theory_id"] for row in result] == ["T1", "T2"]
    first_row = result[0]
    assert first_row["Q1"] == "Yes, but not shown"
    assert first_row["Q2"] == "Yes"
    for question in QUESTION_COLUMNS[2:]:
        assert question in first_row
        assert first_row[question] == ""

    second_row = result[1]
    assert second_row["Q1"] == "No"
    assert second_row["paper_name"] == "Beta study"
