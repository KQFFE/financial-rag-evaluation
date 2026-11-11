from __future__ import annotations

import re
from collections.abc import Mapping

from financial_rag_eval.assistant import Answer, content_terms


def grade_case(case: Mapping[str, object], answer: Answer) -> list[str]:
    issues: list[str] = []
    if not case["answerable"]:
        if not answer.abstained:
            issues.append("Expected an explicit abstention for an unsupported question.")
        expected_response = str(case["expected_response"])
        if answer.text.strip() != expected_response:
            issues.append("Abstention does not match the benchmark's expected response.")
        if answer.citations:
            issues.append("An abstention must not include source citations.")
        return issues

    if answer.abstained:
        return ["Expected a supported answer, but the assistant abstained."]

    expected_facts = case.get("expected_facts", [case.get("expected_answer")])
    for expected_fact in expected_facts:
        fact = str(expected_fact)
        if fact.casefold() not in answer.text.casefold():
            issues.append(f"Answer does not include expected fact: {fact!r}.")

    expected_evidence = [
        {
            "source_id": str(case["source_id"]),
            "passage_id": str(case["passage_id"]),
        },
        *case.get("additional_evidence", []),
    ]
    expected_citations = {
        f"{item['source_id']}#{item['passage_id']}" for item in expected_evidence
    }
    for citation in expected_citations - set(answer.citations):
        issues.append(f"Answer is missing the expected citation [{citation}].")

    cited_passages = [
        item
        for item in answer.retrieved_passages
        if f"{item.source_id}#{item.passage.passage_id}" in answer.citations
    ]
    retrieved_citations = {
        f"{item.source_id}#{item.passage.passage_id}" for item in cited_passages
    }
    for citation in expected_citations - retrieved_citations:
        issues.append(f"Expected cited passage [{citation}] was not retrieved.")

    unexpected_citations = set(answer.citations) - retrieved_citations
    if unexpected_citations:
        issues.append(
            "Answer cites passages that were not retrieved: "
            + ", ".join(sorted(unexpected_citations))
            + "."
        )

    if cited_passages and content_terms(
        re.sub(r"\[[^\]]+\]", "", answer.text)
    ) - set().union(
        *(
            content_terms(
                item.passage.text + " " + item.title + " " + item.publication_date
            )
            for item in cited_passages
        )
    ):
        issues.append("Answer contains terms not supported by its cited passages.")

    return issues
