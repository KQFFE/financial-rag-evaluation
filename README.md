# FinDocEval

FinDocEval is a small evaluation demo for answers grounded in public financial
documents. It is not a full chatbot product, financial advice, or a compliance
certification.

## What it does

The demo answers questions using passages from a small set of bundled public
documents and returns citations in the form `[source_id#passage_id]`. It uses
deterministic keyword overlap; there is no web UI, vector database,
or model call in the local command-line interface (CLI). If retrieval finds no
relevant passage, it returns an explicit "I don't know" without calling the
answerer. When evidence is found, the CLI returns the matching passage itself,
so the demo's answer is an extract rather than a generated summary.

The current benchmark includes the Federal Reserve's September 18 and
November 7, 2024 FOMC statements and Microsoft's FY2024 Form 10-K. Cases cover
rate decisions, a cross-statement comparison, labor-market language,
Microsoft's revenue and net income with reporting period and units, an
unsupported company-revenue question, and requests to reveal hidden
instructions. Source records include URLs, publication dates, and access
dates. The included excerpts are intentionally short; consult the source
terms before redistributing source material.

## How to run

Requires Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Ask a question using the bundled documents:

```powershell
python -m financial_rag_eval "How much revenue did Microsoft report for the fiscal year ended June 30, 2024, in USD millions?"
```

For routine offline checks:

```powershell
python -m pytest
```

The tests use fixed sample responses instead of calling an AI model, so they
run offline without an API key. They check that each benchmark question gets
the expected answer or abstains, and that answers cite passages that support
their claims. Other tests check unsupported questions, made-up claims,
date-specific document retrieval, attempts to reveal hidden instructions,
malicious instructions inside source documents, and saving live-evaluation
results. GitHub Actions runs these tests on every push and pull request.

The grounding checker is deliberately lightweight: it checks expected answer
text, citations, retrieved source passages, and overlapping content terms. It
is not a semantic verifier and cannot establish that all possible claims are
true.

## Optional live-model evaluation

Live evaluation is a separate, opt-in command. It sends the small fixed
benchmark's answerable questions to OpenAI's Chat Completions API using
`gpt-4o-mini` by default. Retrieval-only abstentions do not call the model.
Set the key only in the environment; never put it in source control:

```powershell
$env:OPENAI_API_KEY = "..."
python -m financial_rag_eval.live_evaluate --output evaluations/run.json
```

Set `OPENAI_MODEL` to select another compatible model. The command prints a
case-by-case grade and writes a JSON snapshot containing the model, timestamp,
answers, citations, and grading issues. Live output can vary and may incur API
costs; it is not run by CI. No live-model result is included in this repository
yet. The benchmark is a small demo with cases reviewed by hand; passing it does
not establish general model safety or financial accuracy.
