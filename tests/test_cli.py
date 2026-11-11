import runpy
import sys

import pytest

from financial_rag_eval import cli
from financial_rag_eval.assistant import ABSTENTION_MESSAGE, Answer


def test_cli_prints_answer_and_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys, "argv", ["fin-eval", "What was Microsoft's annual revenue?"]
    )

    assert cli.main() == 0

    output = capsys.readouterr().out
    assert "$245,122 million" in output
    assert "[microsoft-2024-10k#revenue-fy2024]" in output
    assert "Source: Microsoft Corporation 2024 Form 10-K" in output
    assert "https://www.sec.gov/Archives/edgar/" in output


def test_cli_prints_abstention_without_source(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["fin-eval", "What was Nvidia's quarterly revenue?"])

    assert cli.main() == 0

    assert capsys.readouterr().out.strip() == ABSTENTION_MESSAGE


def test_cli_rejects_missing_question(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["fin-eval"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "question" in capsys.readouterr().err


def test_cli_rejects_empty_question(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fin-eval", "  "])

    with pytest.raises(ValueError, match="Question must not be empty"):
        cli.main()


@pytest.mark.parametrize(
    "citation",
    ["unknown#passage", "microsoft-2024-10k#unknown"],
)
def test_cli_rejects_unknown_citation(monkeypatch, capsys, citation) -> None:
    class InvalidAssistant:
        def __init__(self, _documents) -> None:
            pass

        def answer(self, _question: str) -> Answer:
            return Answer("Unverifiable answer.", (citation,), False, ())

    monkeypatch.setattr(cli, "FinancialAssistant", InvalidAssistant)
    monkeypatch.setattr(sys, "argv", ["fin-eval", "question"])

    with pytest.raises(ValueError, match="unknown"):
        cli.main()
    assert capsys.readouterr().out == ""


def test_module_entry_point_propagates_exit_code(monkeypatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("financial_rag_eval", run_name="__main__")

    assert exc_info.value.code == 7
