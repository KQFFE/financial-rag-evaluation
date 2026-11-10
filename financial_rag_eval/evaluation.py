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

    expected_answer = str(case["expected_answer"])
    if expected_answer.casefold() not in answer.text.casefold():
        issues.append(f"Answer does not include expected fact: {expected_answer!r}.")

    source_id = str(case["source_id"])
    passage_id = str(case["passage_id"])
    citation = f"{source_id}#{passage_id}"
    if citation not in answer.citations:
        issues.append(f"Answer is missing the expected citation [{citation}].")

    cited_passages = [
        item
        for item in answer.retrieved_passages
        if f"{item.source_id}#{item.passage.passage_id}" in answer.citations
    ]
    if not any(item.source_id == source_id and item.passage.passage_id == passage_id
               for item in cited_passages):
        issues.append("Expected cited passage was not retrieved.")
    elif content_terms(re.sub(r"\[[^\]]+\]", "", answer.text)) - set().union(
        *(
            content_terms(item.passage.text + " " + item.title)
            for item in cited_passages
        )
    ):
        issues.append("Answer contains terms not supported by its cited passages.")

    return issues
