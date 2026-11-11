import json

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
