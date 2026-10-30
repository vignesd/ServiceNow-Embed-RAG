from enum import Enum


class RetrievalMode(str, Enum):
    """Retrieval strategy for the current user question."""

    FOCUSED = "focused"
    PROCEDURAL = "procedural"


class ResponseDetail(str, Enum):
    """Presentation detail level for the generated answer."""

    CONCISE = "concise"
    DETAILED = "detailed"


PROCEDURAL_PATTERNS: tuple[str, ...] = (
    "how do i",
    "how to",
    "how can i",
    "steps to",
    "steps for",
    "step-by-step",
    "step by step",
    "procedure",
    "process for",
    "guide me",
    "walk me through",
    "detailed guide",
    "full guide",
    "complete guide",
    "full procedure",
    "complete procedure",
    "full requirements",
    "complete requirements",
    "what information do i need",
    "what details do i need",
    "what fields do i need",
)


DETAILED_RESPONSE_PATTERNS: tuple[str, ...] = (
    "detailed guide",
    "detailed steps",
    "step-by-step",
    "step by step",
    "field-by-field",
    "field by field",
    "full guide",
    "complete guide",
    "explain in detail",
    "detailed explanation",
    "all fields",
    "all details",
    "in detail",
)
