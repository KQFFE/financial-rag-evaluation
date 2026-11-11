from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from financial_rag_eval.assistant import FinancialAssistant, load_benchmark, load_documents
from financial_rag_eval.evaluation import grade_case


def save_snapshot(output_path: Path, snapshot: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class OpenAIAnswerer:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def __call__(self, prompt: str) -> str:
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 300,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the optional live-model benchmark and save its results."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the JSON evaluation snapshot.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY to opt in to live-model evaluation.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    assistant = FinancialAssistant(
        load_documents(), OpenAIAnswerer(api_key=api_key, model=model)
    )
    failures = 0
    results: list[dict[str, object]] = []
    for case in load_benchmark():
        answer = assistant.answer(case["question"])
        issues = grade_case(case, answer)
        status = "PASS" if not issues else "FAIL"
        print(f"{status} {case['case_id']}: {answer.text}")
        for issue in issues:
            print(f"  - {issue}")
        failures += bool(issues)
        results.append(
            {
                "case_id": case["case_id"],
                "question": case["question"],
                "answer": answer.text,
                "citations": list(answer.citations),
                "abstained": answer.abstained,
                "issues": issues,
            }
        )
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "case_count": len(results),
        "failure_count": failures,
        "cases": results,
    }
    save_snapshot(args.output, snapshot)
    print(f"Model: {model}; cases: {len(results)}; failures: {failures}")
    print(f"Snapshot saved to: {args.output}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
