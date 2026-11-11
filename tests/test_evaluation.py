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
