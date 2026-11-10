from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from financial_rag_eval.assistant import FinancialAssistant, load_benchmark, load_documents
from financial_rag_eval.evaluation import grade_case


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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY to opt in to live-model evaluation.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    assistant = FinancialAssistant(
        load_documents(), OpenAIAnswerer(api_key=api_key, model=model)
    )
    failures = 0
    for case in load_benchmark():
        answer = assistant.answer(case["question"])
        issues = grade_case(case, answer)
        status = "PASS" if not issues else "FAIL"
        print(f"{status} {case['case_id']}: {answer.text}")
        for issue in issues:
            print(f"  - {issue}")
        failures += bool(issues)
    print(f"Model: {model}; cases: {len(load_benchmark())}; failures: {failures}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
