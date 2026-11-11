from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from financial_rag_eval.assistant import Answer, content_terms


_CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+#[a-z0-9-]+)\]", re.IGNORECASE)
_FULL_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)


def _date_matches(text: str, expected: date) -> bool:
    if "-" in text:
        return text == expected.isoformat()
    month, day, year = text.replace(",", "").split()
    return (
        month.casefold() == expected.strftime("%B").casefold()
        and int(day) == expected.day
        and int(year) == expected.year
    )


def _grade_fact_bindings(
    bindings: list[dict[str, Any]], text: str
) -> list[str]:
    issues: list[str] = []
    lower_text = text.casefold()
    previous_end = 0
    segments: dict[str, list[str]] = {}
    for match in _CITATION_PATTERN.finditer(lower_text):
        segments.setdefault(match.group(1), []).append(
            lower_text[previous_end:match.start()]
        )
        previous_end = match.end()

    claims = re.split(r"\[[^\]]+\]|[;\n]", lower_text)
    for binding in bindings:
        citation = f"{binding['source_id']}#{binding['passage_id']}".casefold()
        other_labels = [
            label.casefold()
            for other in bindings
            if other is not binding
            for label in other["date_labels"]
        ]
        cited_segments = segments.get(citation, [])
        expected_date = date.fromisoformat(binding["date_labels"][0])
        conflicting_claim = any(
            any(
                re.search(r"(?<!\w)" + re.escape(label.casefold()) + r"(?!\w)", claim)
                for label in binding["date_labels"]
            )
            and any(
                other["fact"].casefold() in claim
                for other in bindings
                if other is not binding
            )
            for claim in claims
        )
        if conflicting_claim or not cited_segments or any(
            binding["fact"].casefold() not in segment
            or any(
                not _date_matches(match.group(), expected_date)
                for match in _FULL_DATE_PATTERN.finditer(segment)
            )
            or any(
                re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", segment)
                for label in other_labels
            )
            for segment in cited_segments
        ):
            issues.append(
                f"Expected fact {binding['fact']!r} must be paired with "
                f"its date and citation [{citation}]."
            )
    return issues


def grade_case(case: Mapping[str, Any], answer: Answer) -> list[str]:
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

    bindings = case.get("fact_bindings", [])
    issues.extend(_grade_fact_bindings(bindings, answer.text))
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

    date_labels = " ".join(
        " ".join(binding["date_labels"])
        + f" {date.fromisoformat(binding['date_labels'][0]).day}"
        + f" {date.fromisoformat(binding['date_labels'][0]).year}"
        for binding in bindings
    )
    if cited_passages and content_terms(
        re.sub(r"\[[^\]]+\]", "", answer.text)
    ) - set().union(
        *(
            content_terms(
                item.passage.text + " " + item.title + " "
                + item.publication_date + " " + date_labels
            )
            for item in cited_passages
        )
    ):
        issues.append("Answer contains terms not supported by its cited passages.")

    return issues
