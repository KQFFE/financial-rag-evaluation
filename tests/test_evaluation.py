import pytest

from financial_rag_eval.assistant import (
    ABSTENTION_MESSAGE,
    Answer,
    FinancialAssistant,
    load_benchmark,
    load_documents,
)
from financial_rag_eval.evaluation import grade_case


@pytest.fixture
def case_and_evidence():
    case = next(c for c in load_benchmark() if c["case_id"] == "target-range")
    answer = FinancialAssistant(load_documents()).answer(case["question"])
    return case, answer.retrieved_passages


def test_expected_answer_cannot_be_replaced_by_abstention(case_and_evidence) -> None:
    case, evidence = case_and_evidence
    answer = Answer(ABSTENTION_MESSAGE, (), True, evidence)

    assert grade_case(case, answer) == [
        "Expected a supported answer, but the assistant abstained."
    ]


def test_missing_citation_is_rejected(case_and_evidence) -> None:
    case, evidence = case_and_evidence
    answer = Answer("4-3/4 to 5 percent", (), False, evidence)

    issues = grade_case(case, answer)

    assert any("missing the expected citation" in issue for issue in issues)
    assert any("was not retrieved" in issue for issue in issues)


def test_fabricated_citation_is_rejected_even_with_correct_citation(
    case_and_evidence,
) -> None:
    case, evidence = case_and_evidence
    citation = f"{case['source_id']}#{case['passage_id']}"
    answer = Answer(
        f"4-3/4 to 5 percent [{citation}] [invented#passage]",
        (citation, "invented#passage"),
        False,
        evidence,
    )

    assert any("invented#passage" in issue for issue in grade_case(case, answer))


def test_expected_citation_must_actually_be_retrieved(case_and_evidence) -> None:
    case, _ = case_and_evidence
    citation = f"{case['source_id']}#{case['passage_id']}"
    answer = Answer(f"4-3/4 to 5 percent [{citation}]", (citation,), False, ())

    issues = grade_case(case, answer)

    assert any("was not retrieved" in issue for issue in issues)
    assert any("not retrieved" in issue and "cites" in issue for issue in issues)


def test_abstention_must_not_include_citations() -> None:
    case = next(
        c for c in load_benchmark()
        if c["case_id"] == "unsupported-microsoft-quarterly-revenue"
    )
    answer = Answer(ABSTENTION_MESSAGE, ("invented#passage",), True, ())

    assert "An abstention must not include source citations." in grade_case(
        case, answer
    )


@pytest.mark.parametrize(
    "text",
    [
        "2024-09-18: 4-1/2 to 4-3/4 percent; "
        "2024-11-07: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range] "
        "[fomc-statement-2024-11-07#target-range]",
        "September: 4-3/4 to 5 percent "
        "[fomc-statement-2024-11-07#target-range]; "
        "November: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-09-18#target-range]",
        "November: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "September: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]",
        "September: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "November: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]; "
        "September: 4-1/2 to 4-3/4 percent.",
        "4-3/4 to 5 percent [fomc-statement-2024-09-18#target-range]; "
        "4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]; "
        "4-1/2 to 4-3/4 percent [fomc-statement-2024-09-18#target-range]",
        "September 7, 2024: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "November 18, 2024: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]",
        "2024-09-19: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "2024-11-08: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]",
    ],
)
def test_comparison_rejects_mismatched_facts_dates_and_citations(text) -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "target-range-comparison"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citations = (
        "fomc-statement-2024-09-18#target-range",
        "fomc-statement-2024-11-07#target-range",
    )

    issues = grade_case(case, Answer(text, citations, False, evidence))

    assert any("paired" in issue for issue in issues)


@pytest.mark.parametrize(
    "text",
    [
        "September: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "November: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]",
        "2024-11-07: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]; "
        "2024-09-18: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]",
        "September 18, 2024: 4-3/4 to 5 percent "
        "[fomc-statement-2024-09-18#target-range]; "
        "November 7, 2024: 4-1/2 to 4-3/4 percent "
        "[fomc-statement-2024-11-07#target-range]",
    ],
)
def test_comparison_accepts_correctly_paired_facts(text) -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "target-range-comparison"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citations = (
        "fomc-statement-2024-09-18#target-range",
        "fomc-statement-2024-11-07#target-range",
    )

    assert grade_case(case, Answer(text, citations, False, evidence)) == []


def test_comparison_rejects_missing_fact_citation() -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "target-range-comparison"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    answer = Answer(
        "4-3/4 to 5 percent [fomc-statement-2024-09-18#target-range]",
        ("fomc-statement-2024-09-18#target-range",),
        False,
        evidence,
    )

    assert any("paired" in issue for issue in grade_case(case, answer))


@pytest.mark.parametrize(
    "extra",
    [
        "The target range was 5 percent.",
        "The target range was 5 percent [{citation}].",
        "The target range was 5%.",
    ]
)
def test_target_range_rejects_additional_unsupported_percentage(extra) -> None:
    case = next(c for c in load_benchmark() if c["case_id"] == "target-range")
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citation = f"{case['source_id']}#{case['passage_id']}"
    answer = Answer(
        f"4-3/4 to 5 percent [{citation}]. " + extra.format(citation=citation),
        (citation,),
        False,
        evidence,
    )
    assert grade_case(case, answer)


@pytest.mark.parametrize(
    "extra_claim",
    [
        "122,245 million",
        "122,245",
        "245,122 billion",
        "$245,122",
        "-245,122 million",
        "-$245,122 million",
        "245,122 million for the quarterly period",
        "245,122 million for the fiscal year ended June 30, 2023",
        "245,122 million for the fiscal year ended June 24, 2024",
        "245,122 million for the fiscal year ended June 24, 2030",
    ],
)
def test_financial_amounts_reject_unsupported_claims(extra_claim) -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "microsoft-revenue-fy2024"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citation = f"{case['source_id']}#{case['passage_id']}"
    answer = Answer(
        f"245,122 million [{citation}]; {extra_claim} [{citation}]",
        (citation,),
        False,
        evidence,
    )

    assert any("financial" in issue.casefold() for issue in grade_case(case, answer))


def test_financial_amount_must_be_supported_by_its_own_citation() -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "microsoft-revenue-fy2024"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    revenue = "microsoft-2024-10k#revenue-fy2024"
    income = "microsoft-2024-10k#net-income-fy2024"
    assert income in {
        f"{item.source_id}#{item.passage.passage_id}" for item in evidence
    }
    answer = Answer(
        f"245,122 million [{revenue}]; 245,122 million [{income}]",
        (revenue, income),
        False,
        evidence,
    )

    assert any("financial" in issue.casefold() for issue in grade_case(case, answer))


def test_uncited_additional_financial_amount_is_rejected() -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "microsoft-revenue-fy2024"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citation = f"{case['source_id']}#{case['passage_id']}"
    answer = Answer(
        f"245,122 million [{citation}]; 122,245 million.",
        (citation,),
        False,
        evidence,
    )

    assert any("financial" in issue.casefold() for issue in grade_case(case, answer))


@pytest.mark.parametrize(
    "text",
    [
        "245,122 million",
        "$245,122 million for the fiscal year ended June 30, 2024",
        "245,122 million for the fiscal year ended June 30, 2024",
    ],
)
def test_supported_financial_amount_is_accepted(text) -> None:
    case = next(
        c for c in load_benchmark() if c["case_id"] == "microsoft-revenue-fy2024"
    )
    evidence = FinancialAssistant(load_documents()).answer(
        case["question"]
    ).retrieved_passages
    citation = f"{case['source_id']}#{case['passage_id']}"
    # Keep the benchmark's exact expected wording as well as the variant.
    answer = Answer(
        f"245,122 million [{citation}]; {text} [{citation}]",
        (citation,),
        False,
        evidence,
    )

    assert grade_case(case, answer) == []
