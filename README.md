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
answerer. When evidence is found, the no-model CLI returns up to three
retrieved passages with citations. This lets comparison questions show evidence
from more than one document, but the output is an extract rather than a
generated summary and may include extra passages. If a question explicitly
asks for quarterly or annual results, passages with a different reporting
period are excluded; if no matching passage exists, the assistant abstains.
For explicitly stated dates such as `September 18, 2024`, `November 2024`,
or `2024-09-18`, retrieval requires matching dates in the selected evidence.
When an annual period is requested, the date must appear in the passage
rather than merely being the document's publication date. If any recognized
requested date is missing from the selected passages, it abstains without
calling the model. The filter also recognizes four-digit years and `FY2024`
or `FY24` (two-digit `FY` years are interpreted as 2000–2099). For common
possessive forms such as `Nvidia's`, and some capitalized company-name forms,
retrieval also requires the named entity to occur in the passage or document
title. These are simple text filters, not an understanding of the question:
other date and company-name forms may be missed, and a contextual date or
possessive can cause an unnecessary abstention.

The current benchmark includes the Federal Reserve's September 18 and
November 7, 2024 FOMC statements and Microsoft's FY2024 Form 10-K. Cases cover
rate decisions, a cross-statement comparison, labor-market language,
Microsoft's revenue and net income with reporting period and units, an
unsupported company-revenue question, other unsupported date-specific
questions, and requests to
reveal hidden instructions. Source records include URLs, publication dates,
and access dates. Passages are labeled as short Federal Reserve excerpts or
Microsoft financial-fact paraphrases; the latter are not direct quotations.

### Source reuse

Assessment reviewed November 11, 2026:

- The FOMC passages are short excerpts attributed to the Board-hosted
  statements. [17 U.S.C. § 105](https://www.govinfo.gov/content/pkg/USCODE-2023-title17/html/USCODE-2023-title17-chap1-sec105.htm)
  excludes U.S. government works from U.S. copyright protection, but being
  hosted by the Board alone does not establish that status. The
  [Board's website policies](https://www.federalreserve.gov/website-linking-policies.htm)
  also warn that linked third-party material may have separate restrictions.
  Authorship and any applicable redistribution rights for these statements
  have not been independently confirmed.
- The Microsoft passages are project-written paraphrases of two figures from
  its SEC-filed annual report, not direct quotations. The
  [U.S. Copyright Office](https://www.copyright.gov/help/faq/faq-protect.html)
  explains that copyright does not protect facts, although expression can be
  protected. SEC hosting does not make the third-party filing government-authored
  or establish unrestricted reuse of the filing.

This is a limited assessment, not legal advice or a definitive permission
determination; check applicable rights before further redistribution.

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

For the same statement and branch coverage report used in CI:

```powershell
python -m coverage run -m pytest
python -m coverage report -m --fail-under=100
```

The test dependencies are pytest 8 or 9 and coverage.py 7.6 or newer (below
version 8), as specified in the project configuration.

The tests use fixed sample responses instead of calling an AI model, so they
run offline without an API key. They check that each benchmark question gets
the expected answer or abstains, and that answers cite passages that support
their claims. The benchmark is also run through the actual no-model retrieval
and response path. Other tests check unsupported questions, made-up claims,
reporting-period mismatches, date-specific document retrieval, attempts to
reveal hidden instructions, malicious instructions inside source documents,
financial amounts paired with the wrong values, units, periods, or citations,
alternate forms of the injected false revenue, model abstention with
insufficient evidence, and saving live-evaluation results. They also exercise
the CLI and the full
live-evaluation command with a simulated provider, including failure paths;
they do not call the network. GitHub Actions runs these tests on every push
and pull request and requires 100% combined statement and branch coverage
of the application package. Coverage measures executed code, not whether
every possible input, claim, or attack has been tested.

The latest local run on Python 3.14.7 passed 118 tests with 100% combined
statement and branch coverage. This is an offline code-coverage result, not a
measure of real-model answer quality.

The grounding checker is deliberately lightweight: it checks expected answer
text, citations, retrieved source passages, and overlapping content terms. For
the rate comparison, each rate must occur before its own source citation; the
checker also catches swapped month labels, incorrect full dates, and some
contradictory claims. For financial amounts written with a dollar sign,
`USD`, a comma, or a magnitude such as `million`, it checks the amount and
unit against the passage in the following citation. Recognized percentage
values and ranges are also checked against their following citation, including
additional claims after an otherwise correct answer. In the same claim, it
also checks stated annual or quarterly periods, years, and full dates against
that passage. Amounts without a following citation are rejected.

These are conservative text checks: correct answers with different wording,
formats, or citation placement may be rejected. Amounts and rates written in words or
without a recognized currency, comma, or unit may be missed, as can incorrect
relationships between otherwise supported facts. The checker is not a
semantic verifier and cannot establish that all possible claims are true.

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
answers, citations, and grading issues. The snapshot also embeds the loaded
benchmark cases, document data, system instructions, and SHA-256 digests of
canonical JSON for the cases and documents. At startup it records SHA-256
hashes of the package's Python source files and, when available, the Git
commit and whether the whole working tree has changes. A missing checkout or
failed Git command is recorded in `code.git_error`; file hashes still identify
the on-disk source files. A dirty flag does not preserve those changes: keep
the corresponding source separately if the run needs to be reproduced. These
hashes describe files on disk at startup, so avoid editing code during a run.
Live output can vary and may incur API costs; it is not run by
CI. No live-model result is included in this repository yet. The live command
sends application instructions as a system message and
questions plus retrieved passages as a separate user message. It also runs two
extra safety probes: one asks the model to reveal a system-message canary, and
another embeds conflicting instructions and a canary in a synthetic retrieved
passage. These checks report obvious canary leakage or the injected false
revenue, including recognized numeric equivalents such as `999 million` and
`$999,000,000`. The false-revenue check flags a mention even if the answer
rejects it, and can miss unrecognized formats or paraphrases. These probes make
two additional model requests and do not prove the model is secure. Two
additional controlled checks send the model a real Microsoft annual-revenue
passage alongside questions about another company or a quarterly result. They
call the model directly with that insufficient evidence rather than exercising
retrieval, and pass only when it returns the exact abstention with nothing
appended. These checks make two more model requests. Review saved answers and
grades manually: the checks are deliberately narrow and may reject reasonable
alternative wording.

The benchmark is a small demo with cases reviewed by hand; passing it does not
establish general model safety or financial accuracy.

If OpenAI returns HTTP 429, the command reports recognized credit, spending
limit, or request-rate errors without displaying the provider's raw error
message. For credit or quota errors, check
[API billing](https://platform.openai.com/settings/organization/billing/overview)
and your organization or project limits; ChatGPT subscriptions are billed
separately. For a request-rate limit, wait before retrying. Unknown 429 errors
require checking both possibilities. The command does not retry automatically.
