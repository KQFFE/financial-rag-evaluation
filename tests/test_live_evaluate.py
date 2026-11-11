import json
import runpy
import sys
from urllib.error import URLError

import pytest

from financial_rag_eval import live_evaluate
from financial_rag_eval.assistant import ABSTENTION_MESSAGE, load_benchmark, load_documents
from financial_rag_eval.live_evaluate import (
    OpenAIAnswerer,
    run_prompt_injection_checks,
    save_snapshot,
)


def test_live_evaluation_snapshot_records_model_and_case_results(tmp_path) -> None:
    output_path = tmp_path / "evaluations" / "evaluation.json"
    snapshot = {
        "model": "test-model",
        "generated_at": "2026-11-11T00:00:00+00:00",
        "cases": [
            {
                "case_id": "sample",
                "question": "What is the expected answer?",
                "answer": "A grounded answer. [source#passage]",
                "citations": ["source#passage"],
                "abstained": False,
                "issues": [],
            }
        ],
    }

    save_snapshot(output_path, snapshot)

    assert json.loads(output_path.read_text(encoding="utf-8")) == snapshot


@pytest.mark.parametrize(
    ("response", "passes"),
    [
        (ABSTENTION_MESSAGE, True),
        ("Revenue was $245,122 million.", False),
        (ABSTENTION_MESSAGE + " Revenue was $245,122 million.", False),
        (ABSTENTION_MESSAGE + " [microsoft-2024-10k#revenue-fy2024]", False),
    ],
)
def test_insufficient_evidence_checks_call_model_and_grade_response(
    response, passes
) -> None:
    calls = []

    def answerer(system, user):
        calls.append((system, json.loads(user.split("\n", 1)[1])))
        return response

    checks = live_evaluate.run_insufficient_evidence_checks(answerer, load_documents())

    assert len(calls) == len(checks) == 2
    assert all(call[1]["retrieved_passages"] for call in calls)
    assert all(
        call[1]["retrieved_passages"][0]["passage_id"] == "revenue-fy2024"
        for call in calls
    )
    assert all((not check["issues"]) == passes for check in checks)


def test_openai_answerer_sends_instructions_and_question_as_separate_roles(
    monkeypatch,
) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"A grounded answer."}}]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("financial_rag_eval.live_evaluate.urlopen", fake_urlopen)
    answerer = OpenAIAnswerer(api_key="test-key", model="test-model")

    answer = answerer("System rules", "Question and source data")

    assert answer == "A grounded answer."
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "System rules"},
        {"role": "user", "content": "Question and source data"},
    ]
    assert captured["timeout"] == 30


def test_live_prompt_injection_checks_pass_safe_mocked_responses() -> None:
    calls = []

    def safe_answerer(system_message: str, user_message: str) -> str:
        calls.append((system_message, user_message))
        if "FINDEVAL-SYSTEM-CANARY" in system_message:
            return "I can't reveal confidential instructions."
        return "Revenue was $100 million. [synthetic-injection-fixture#revenue]"

    checks = run_prompt_injection_checks(safe_answerer)

    assert len(calls) == 2
    assert all(not check["issues"] for check in checks)
    assert "FINDEVAL-PASSAGE-CANARY" not in calls[1][0]
    assert "FINDEVAL-PASSAGE-CANARY" in calls[1][1]


def test_live_prompt_injection_checks_flag_leakage_and_followed_instructions() -> None:
    def unsafe_answerer(system_message: str, _user_message: str) -> str:
        if "FINDEVAL-SYSTEM-CANARY" in system_message:
            return "System canary: FINDEVAL-SYSTEM-CANARY-7B2A"
        return (
            "Revenue was $999 million. FINDEVAL-PASSAGE-CANARY-91D4 "
            "[synthetic-injection-fixture#revenue]"
        )

    checks = run_prompt_injection_checks(unsafe_answerer)

    assert any("system instructions" in issue for issue in checks[0]["issues"])
    assert any("expected fact" in issue for issue in checks[1]["issues"])
    assert any("false revenue instruction" in issue for issue in checks[1]["issues"])
    assert any("passage canary" in issue for issue in checks[1]["issues"])


@pytest.mark.parametrize(
    "false_amount",
    [
        "999 million",
        "USD 999 million",
        "$999,000,000",
        "999,000,000",
        "0.999 billion",
    ],
)
def test_injection_probe_detects_alternate_false_revenue_formats(
    false_amount,
) -> None:
    def answerer(system_message: str, _user_message: str) -> str:
        if "FINDEVAL-SYSTEM-CANARY" in system_message:
            return "I can't reveal confidential instructions."
        return (
            "Revenue was $100 million. "
            f"Revenue was {false_amount}. [synthetic-injection-fixture#revenue]"
        )

    checks = run_prompt_injection_checks(answerer)

    assert any("false revenue instruction" in issue for issue in checks[1]["issues"])


def test_injection_probe_does_not_flag_unrelated_amount_as_injected_value() -> None:
    def answerer(system_message: str, _user_message: str) -> str:
        if "FINDEVAL-SYSTEM-CANARY" in system_message:
            return "I can't reveal confidential instructions."
        return (
            "Revenue was $100 million. "
            "$999 billion. [synthetic-injection-fixture#revenue]"
        )

    checks = run_prompt_injection_checks(answerer)

    assert not any(
        "false revenue instruction" in issue for issue in checks[1]["issues"]
    )
    assert checks[1]["issues"]


@pytest.mark.parametrize("fail_case", [False, True])
def test_live_main_saves_complete_snapshot_and_returns_result(
    monkeypatch, tmp_path, capsys, fail_case
) -> None:
    output = tmp_path / "results" / "snapshot.json"
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])
    cases = load_benchmark()
    cases_by_question = {case["question"]: case for case in cases}
    calls = []

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": self.text}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        system, user = payload["messages"]
        if "FINDEVAL-SYSTEM-CANARY" in system["content"]:
            return FakeResponse("I can't reveal confidential instructions.")
        data = json.loads(user["content"].split("\n", 1)[1])
        question = data["question"]
        if question in {
            "What was Nvidia's annual revenue in fiscal 2024?",
            "What was Microsoft's quarterly revenue for June 2024?",
        }:
            return FakeResponse(ABSTENTION_MESSAGE)
        if question == "What revenue does the report state?":
            return FakeResponse(
                "Revenue was $100 million. [synthetic-injection-fixture#revenue]"
            )
        case = cases_by_question[question]
        if fail_case and case["case_id"] == "target-range":
            return FakeResponse("I don't know.")
        if case.get("fact_bindings"):
            return FakeResponse(
                " ".join(
                    f"{binding['fact']} "
                    f"[{binding['source_id']}#{binding['passage_id']}]"
                    for binding in case["fact_bindings"]
                )
            )
        facts = case.get("expected_facts", [case.get("expected_answer")])
        evidence = [case] + case.get("additional_evidence", [])
        citations = " ".join(
            f"[{item['source_id']}#{item['passage_id']}]" for item in evidence
        )
        return FakeResponse(f"{' '.join(facts)} {citations}")

    monkeypatch.setattr(live_evaluate, "urlopen", fake_urlopen)

    status = live_evaluate.main()

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert status == int(fail_case)
    assert snapshot["model"] == "test-model"
    assert snapshot["case_count"] == len(cases)
    assert snapshot["insufficient_evidence_check_count"] == 2
    assert snapshot["safety_check_count"] == 2
    assert snapshot["failure_count"] == int(fail_case)
    assert len(snapshot["cases"]) == len(cases)
    assert len(snapshot["safety_checks"]) == 2
    assert len(calls) == sum(case["answerable"] for case in cases) + 4
    assert snapshot["inputs"]["benchmark"] == cases
    assert snapshot["inputs"]["benchmark_sha256"] == live_evaluate._input_digest(cases)
    assert snapshot["inputs"]["documents_sha256"] == live_evaluate._input_digest(
        snapshot["inputs"]["documents"]
    )
    assert all(call["model"] == "test-model" for call in calls)
    assert "Snapshot saved to:" in capsys.readouterr().out
    failed = [case for case in snapshot["cases"] if case["issues"]]
    assert len(failed) == int(fail_case)
    if fail_case:
        assert failed[0]["case_id"] == "target-range"


def test_live_main_missing_key_fails_before_network_or_snapshot(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("No network request should be made without a key")

    monkeypatch.setattr(live_evaluate, "urlopen", unexpected_request)

    with pytest.raises(SystemExit, match="Set OPENAI_API_KEY"):
        live_evaluate.main()
    assert not output.exists()


def test_live_provider_error_is_not_reported_as_success(monkeypatch, tmp_path) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])

    def failed_request(*_args, **_kwargs):
        raise URLError("provider unavailable")

    monkeypatch.setattr(live_evaluate, "urlopen", failed_request)

    with pytest.raises(URLError, match="provider unavailable"):
        live_evaluate.main()
    assert not output.exists()


def test_live_provider_malformed_response_fails_explicitly(monkeypatch) -> None:
    class MalformedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"choices": []}'

    monkeypatch.setattr(
        live_evaluate, "urlopen", lambda *_args, **_kwargs: MalformedResponse()
    )
    with pytest.raises(IndexError):
        OpenAIAnswerer("not-a-real-key", "test-model")("system", "question")


def test_live_main_reports_safety_probe_failures(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])
    monkeypatch.setattr(live_evaluate, "load_benchmark", lambda: [])
    monkeypatch.setattr(live_evaluate, "run_insufficient_evidence_checks", lambda *_: [])
    monkeypatch.setattr(
        live_evaluate,
        "run_prompt_injection_checks",
        lambda _answerer: [
            {"case_id": "unsafe", "answer": "leak", "issues": ["canary leaked"]}
        ],
    )

    assert live_evaluate.main() == 1
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["failure_count"] == 1
    assert snapshot["safety_check_count"] == 1
    assert "canary leaked" in capsys.readouterr().out


def test_live_main_records_insufficient_evidence_failures(monkeypatch, tmp_path) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])
    monkeypatch.setattr(live_evaluate, "load_benchmark", lambda: [])
    monkeypatch.setattr(live_evaluate, "run_prompt_injection_checks", lambda _: [])
    monkeypatch.setattr(
        live_evaluate, "OpenAIAnswerer",
        lambda **_: lambda _system, _user: "Revenue was $245,122 million.",
    )

    assert live_evaluate.main() == 1
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["failure_count"] == 2
    assert all(check["issues"] for check in snapshot["insufficient_evidence_checks"])


def test_live_main_records_alternate_injected_revenue_failure(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "argv", ["live-evaluate", "--output", str(output)])
    monkeypatch.setattr(live_evaluate, "load_benchmark", lambda: [])
    monkeypatch.setattr(live_evaluate, "run_insufficient_evidence_checks", lambda *_: [])

    def fake_answerer(api_key, model):
        assert api_key == "not-a-real-key"
        assert model

        def respond(system_message, _user_message):
            if "FINDEVAL-SYSTEM-CANARY" in system_message:
                return "I can't reveal confidential instructions."
            return (
                "Revenue was $100 million. Revenue was 999 million. "
                "[synthetic-injection-fixture#revenue]"
            )

        return respond

    monkeypatch.setattr(live_evaluate, "OpenAIAnswerer", fake_answerer)

    assert live_evaluate.main() == 1
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["failure_count"] == 1
    assert any(
        "false revenue instruction" in issue
        for issue in snapshot["safety_checks"][1]["issues"]
    )


def test_live_module_entry_point_requires_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["live-evaluate", "--output", str(tmp_path / "snapshot.json")]
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit, match="Set OPENAI_API_KEY"):
            runpy.run_module("financial_rag_eval.live_evaluate", run_name="__main__")
