import json

from financial_rag_eval.live_evaluate import save_snapshot


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
