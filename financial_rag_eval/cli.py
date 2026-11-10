from __future__ import annotations

import argparse

from financial_rag_eval.assistant import FinancialAssistant, load_documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer a question using the bundled public financial document."
    )
    parser.add_argument("question", help="Question to answer from the source passages.")
    args = parser.parse_args()

    documents = load_documents()
    answer = FinancialAssistant(documents).answer(args.question)
    print(answer.text)
    for citation in answer.citations:
        source_id, passage_id = citation.split("#", maxsplit=1)
        for document in documents:
            if document.source_id == source_id:
                print(f"Source: {document.title} ({passage_id}) - {document.url}")
                break
    return 0
