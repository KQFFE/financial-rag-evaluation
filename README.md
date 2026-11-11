# FinDocEval

FinDocEval is a small evaluation demo for answers grounded in public financial
documents. It is not a full chatbot product, financial advice, or a compliance
certification.

The benchmark uses short passages from the Federal Reserve's September 18 and
November 7, 2024 FOMC statements, plus Microsoft's FY2024 Form 10-K. Cases
include a comparison across the FOMC statements, company financial facts with
their reporting period and units, and unsupported questions. Each source
record includes its URL, publication date, and access date. The passages are
intentionally short; consult the source terms before redistributing source
material.

## Setup

Requires Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Offline checks

```powershell
python -m pytest
```

These deterministic tests use stubbed answerers and make no model or network
calls. They run every benchmark case against a mocked answerer and check its
expected fact or abstention, cited source passages, and grounding. Additional
cases cover unsupported claims, user requests for hidden instructions, and
malicious instructions embedded in retrieved text. The demo retrieval is
simple keyword overlap, not a semantic search system.

## Try the assistant

```powershell
python -m financial_rag_eval "What target range did the FOMC set on September 18, 2024?"
```

The local demo returns the best-matching source passage with its citation; it
does not call an LLM.

## Optional live-model evaluation

Live evaluation is a separate, opt-in command. It sends the small fixed
benchmark to OpenAI's Chat Completions API using `gpt-4o-mini` by default.
Set the key only in the environment; never put it in source control:

```powershell
$env:OPENAI_API_KEY = "..."
python -m financial_rag_eval.live_evaluate --output evaluations/fy2024.json
```

Set `OPENAI_MODEL` to select another compatible model. Live results can vary
and may incur API costs. The command records the model, timestamp, answers,
citations, and grading issues in the requested JSON snapshot; it is not part of
CI. The offline suite remains the CI gate. The benchmark is a small hand-reviewed
demo and passing it does not establish general model safety or financial
accuracy.
