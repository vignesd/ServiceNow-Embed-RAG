import pytest

from chat.agent import (
    classify_response_detail,
    classify_retrieval_mode,
)
from chat.schema import ResponseDetail, RetrievalMode


@pytest.mark.parametrize(
    "query",
    [
        "How do I create a CIBC Login ID?",
        "How to create Windows Group request",
        "How can I request a Windows Group?",
        "What are the steps to request a CIBC Login ID?",
        "Steps for creating a Windows Group request",
        "Guide me through the CIBC Login ID request",
        "Walk me through the Windows Group request",
        "What is the procedure for creating a CIBC Login ID?",
        "What is the process for a Windows Group request?",
        "What information do I need for a CIBC Login ID?",
        "What details do I need for a Windows Group request?",
        "What fields do I need for a CIBC Login ID request?",
        "Give me the complete requirements for a Windows Group request",
    ],
)
def test_procedural_queries_route_to_guide_expansion(query: str) -> None:
    assert classify_retrieval_mode(query) is RetrievalMode.PROCEDURAL


@pytest.mark.parametrize(
    "query",
    [
        "What is the SLA for a CIBC Login ID?",
        "Who approves the Windows Group request?",
        "Is Telephone Number mandatory for a CIBC Login ID?",
        "What System/Environment options are available?",
        "What is the escalation contact?",
        "Is Employee Number required?",
        "What is the catalog URL?",
        "Explain the SLA in detail",
    ],
)
def test_focused_queries_do_not_expand_guide(query: str) -> None:
    assert classify_retrieval_mode(query) is RetrievalMode.FOCUSED


@pytest.mark.parametrize(
    "question",
    [
        "How to create Windows Group request",
        "How do I create a CIBC Login ID?",
        "What is the SLA for a CIBC Login ID?",
        "Who approves this request?",
        "What System/Environment options are available?",
    ],
)
def test_normal_questions_default_to_concise(question: str) -> None:
    assert classify_response_detail(question) is ResponseDetail.CONCISE


@pytest.mark.parametrize(
    "question",
    [
        "Give me a detailed guide to create a Windows Group request",
        "Give me detailed steps for a CIBC Login ID request",
        "Give me a step-by-step guide for a Windows Group request",
        "Show me the field-by-field guide",
        "Give me the full guide for CIBC Login ID",
        "Give me the complete guide for Windows Group request",
        "Explain the CIBC Login ID request in detail",
        "Give me a detailed explanation of the Windows Group request",
        "Show me all fields for the CIBC Login ID request",
        "Give me all details for the Windows Group request",
    ],
)
def test_explicit_detail_requests_route_to_detailed(question: str) -> None:
    assert classify_response_detail(question) is ResponseDetail.DETAILED


@pytest.mark.parametrize(
    ("question", "expected_retrieval", "expected_detail"),
    [
        (
            "How to create Windows Group request",
            RetrievalMode.PROCEDURAL,
            ResponseDetail.CONCISE,
        ),
        (
            "Give me a detailed guide to create Windows Group request",
            RetrievalMode.PROCEDURAL,
            ResponseDetail.DETAILED,
        ),
        (
            "Walk me through the Windows Group request step by step",
            RetrievalMode.PROCEDURAL,
            ResponseDetail.DETAILED,
        ),
        (
            "What is the Windows Group SLA?",
            RetrievalMode.FOCUSED,
            ResponseDetail.CONCISE,
        ),
        (
            "Explain the Windows Group SLA in detail",
            RetrievalMode.FOCUSED,
            ResponseDetail.DETAILED,
        ),
        (
            "Which Type of Request options are available?",
            RetrievalMode.FOCUSED,
            ResponseDetail.CONCISE,
        ),
    ],
)
def test_retrieval_and_response_detail_are_independent(
    question: str,
    expected_retrieval: RetrievalMode,
    expected_detail: ResponseDetail,
) -> None:
    assert classify_retrieval_mode(question) is expected_retrieval
    assert classify_response_detail(question) is expected_detail


@pytest.mark.parametrize(
    "query",
    [
        "  HOW   TO   CREATE   WINDOWS GROUP REQUEST  ",
        "How\nDo\tI create a CIBC Login ID?",
        "STEP-BY-STEP Windows Group request",
    ],
)
def test_classifiers_are_case_and_whitespace_tolerant(query: str) -> None:
    assert classify_retrieval_mode(query) is RetrievalMode.PROCEDURAL


def test_detailed_word_alone_does_not_force_procedural_retrieval() -> None:
    question = "Explain the SLA in detail"

    assert classify_retrieval_mode(question) is RetrievalMode.FOCUSED
    assert classify_response_detail(question) is ResponseDetail.DETAILED
